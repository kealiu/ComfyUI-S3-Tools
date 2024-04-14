from .nodes.node_S3 import *

NODE_CLASS_MAPPINGS = {
    "Save Image To S3": SaveImageToS3,
    "Load Image From S3": LoadImageFromS3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Image To S3": "💾 Save Your Image to S3",
    "Load Image From S3": "💾 Load Your Image From S3"
}
