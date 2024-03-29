from typing import Dict, Optional, Sequence

import torch
from docarray import DocumentArray
from jina import Executor, requests
from transformers import CLIPModel, CLIPTokenizer

import warnings

class CLIPTextEncoder(Executor):
    """Encode text into embeddings using the CLIP model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'openai/clip-vit-base-patch32',
        base_tokenizer_model: Optional[str] = None,
        max_length: int = 77,
        device: str = 'cpu',
        access_paths: str = '@r',
        traversal_paths: Optional[str] = None,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        :param pretrained_model_name_or_path: Can be either:
            - A string, the model id of a pretrained CLIP model hosted
                inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
            - A path to a directory containing model weights saved, e.g., ./my_model_directory/
        :param base_tokenizer_model: Base tokenizer model.
            Defaults to ``pretrained_model_name_or_path`` if None
        :param max_length: Max length argument for the tokenizer.
            All CLIP models use 77 as the max length
        :param device: Pytorch device to put the model on, e.g. 'cpu', 'cuda', 'cuda:1'
        :param access_paths: Default traversal paths for encoding, used if
            the traversal path is not passed as a parameter with the request.
        :param traversal_paths: please use access_paths
        :param batch_size: Default batch size for encoding, used if the
            batch size is not passed as a parameter with the request.
        """
        super().__init__(*args, **kwargs)
        if traversal_paths is not None:
            self.access_paths = traversal_paths
            warnings.warn("'traversal_paths' will be deprecated in the future, please use 'access_paths'.",
                          DeprecationWarning,
                          stacklevel=2)
        else:
            self.access_paths = access_paths
        self.batch_size = batch_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.max_length = max_length

        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_tokenizer_model)
        self.model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path)
        self.model.eval().to(device)

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """
        Encode all documents with the `text` attribute and store the embeddings in the
        `embedding` attribute.

        :param docs: DocumentArray containing the Documents to be encoded
        :param parameters: A dictionary that contains parameters to control encoding.
            The accepted keys are ``access_paths`` and ``batch_size`` - in their
            absence their corresponding default values are used.
        """

        for docs_batch in DocumentArray(
            filter(
                lambda x: bool(x.text),
                docs[parameters.get('access_paths', self.access_paths)],
            )
        ).batch(batch_size=parameters.get('batch_size', self.batch_size)) :

            text_batch = docs_batch.texts

            with torch.inference_mode():
                input_tokens = self._generate_input_tokens(text_batch)
                embeddings = self.model.get_text_features(**input_tokens).cpu().numpy()
                for doc, embedding in zip(docs_batch, embeddings):
                    doc.embedding = embedding

    def _generate_input_tokens(self, texts: Sequence[str]):

        input_tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )
        input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        return input_tokens
