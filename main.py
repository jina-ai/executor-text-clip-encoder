from jina import Flow

from clip_text import CLIPTextEncoder


if __name__ == "__main__":

    f = Flow().add(uses=CLIPTextEncoder)

    with f:
        pass


