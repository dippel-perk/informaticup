from PIL import Image

img = Image.open("../road.jpg")
new_img = img.resize((64, 64))
new_img.save("../road_resized.jpg")
