from PIL import Image
import io

from transformers import AutoProcessor, AutoModel

device = "cuda"
processor = AutoProcessor.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K")

# 打开图像文件
image_path = "/home/wenhesun/Plum_changed/prompt_0_images_1.png"
image = Image.open(image_path)

# 获取图像的大小（宽度 x 高度）
image_size = image.size

# 输出图像的大小
print("Image size:", image_size)
print("type(image):", type(image))

image_inputs = processor(
    images=image,
    padding=True,
    truncation=True,
    max_length=77,
    return_tensors="pt",
).to(device)

print("type(image_inputs):", type(image_inputs))

# 从 BatchEncoding 中提取图像的二进制数据
image_binary_data = image_inputs["pixel_values"][0].cpu().numpy()

# 将二进制数据还原为 PngImageFile 对象
png_image = Image.open(io.BytesIO(image_binary_data.tobytes()))

# 检查类型
print(type(png_image))