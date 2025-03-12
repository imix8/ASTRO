from pygoogle_image import image as pi

# List of trash items you want to detect
keywords_list = [
    "plastic bag",
    "aluminum can",
    "plastic bottle",
    "paper cup",
    "glass bottle"
]

# Download a limited number of images for each keyword
for keyword in keywords_list:
    print(f"Downloading images for: {keyword}")
    pi.download(keywords=keyword, limit=10)