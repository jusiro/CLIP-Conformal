import torch
import clip

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Adapter(torch.nn.Module):
    def __init__(self, clip_model, classnames, adapter="ZS", templates=None, clip_id=None):
        super().__init__()

        # Init
        self.adapt_strategy = adapter
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad = False
        self.templates = templates
        self.clip_id = clip_id  # ID for family of pre-trained model

        # Set templates
        if templates is None:
            self.templates = ["a photo of a {}."]

        # Set strategy for classifier head initialization
        self.init = "random" if ("RI" in adapter) else "zero_shot"

        # Obtain class prototypes
        text_embeddings_avg, text_embeddings = self.get_text_prototypes(clip_model, classnames, self.templates,
                                                                        clip_id=self.clip_id)
        self.text_embeddings_avg, self.text_embeddings = text_embeddings_avg.cpu(), text_embeddings.cpu()

        # Set classifier
        self.adapter = LinearProbeHead(text_embeddings_avg, self.logit_scale, init=self.init)

        # move to device
        self.to(device).float()

    def forward(self, x):

        # Forward classifier
        out = self.adapter(x)

        return out

    def reset(self):

        # Set classifier
        self.adapter = LinearProbeHead(self.text_embeddings_avg, self.logit_scale, init=self.init)

        # move to device
        self.to(device).float()

    @staticmethod
    def get_text_prototypes(clip_model, classnames, templates, clip_id="CLIP"):
        clip_model.eval()

        if clip_id.split("-")[0] == "MetaCLIP":
            from transformers import AutoProcessor
            # Name to Transformers Huggingface ID.
            name_id_match = {"MetaCLIP-ViT-B/16": "facebook/metaclip-b16-fullcc2.5b",
                             "MetaCLIP-ViT-H/14": "facebook/metaclip-h14-fullcc2.5b"}
            # Load tokenizer
            tokenizer = AutoProcessor.from_pretrained(name_id_match[clip_id]).tokenizer

        print("Extracting text prototypes from class names...")
        with torch.no_grad():
            text_embeddings = []
            for text in classnames:
                if clip_id.split("-")[0] == "CLIP":
                    tokens = clip.tokenize(
                        [template.format(text.replace('_', ' ')) for template in templates]).to(device)
                    prototype = clip_model.encode_text(tokens.to(device))
                elif clip_id.split("-")[0] == "MetaCLIP":
                    # Prepare tokens
                    tokens = tokenizer([template.format(text.replace('_', ' ')) for template in templates],
                                       return_tensors="pt", padding=True).to(device)
                    # Get text prototype
                    prototype = clip_model.get_text_features(input_ids=tokens["input_ids"].to(device),
                                                             attention_mask=tokens["attention_mask"].to(device))
                # Add prototypes
                text_embeddings.append(prototype)

        text_embeddings = torch.stack(text_embeddings)
        text_embeddings_avg = text_embeddings.mean(1)
        return text_embeddings_avg, text_embeddings


class LinearProbeHead(torch.nn.Module):
    def __init__(self, zero_shot_prot, logit_scale, init="zero_shot"):
        super().__init__()
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False
        self.init = init
        self.zero_shot_prot = zero_shot_prot.clone()

        if init == "zero_shot":
            self.prototypes = zero_shot_prot.clone()
        else:
            self.prototypes = torch.nn.init.kaiming_normal_(torch.empty(zero_shot_prot.shape))

        # Trainable parameters
        self.prototypes = torch.nn.Parameter(self.prototypes)

        # Keep temperature scaling as in pre-training
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False

    def forward(self, features):

        # Get trained prototype
        prototypes = self.prototypes.to(device)

        # Unit hypersphere normalization
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        return logits