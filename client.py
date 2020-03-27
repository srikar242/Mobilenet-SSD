import asyncio
import aiohttp
import os
from aiohttp import FormData

# Change the IP in the following with the AI server IP.
# The API runs on port 5000.
url = "http://192.168.43.157:5000/api/v1/upload"
#url = "http://0.0.0.0:5000/api/v1/upload"

async def post_data(image, image_filename):
    """
    image: binary image data (opened file)
    image_filename: string, filename. It can be None.
    """
    global url
    data = FormData()
    data.add_field('file',
                   image,
                   filename=image_filename,
                   content_type="image/jpg; multipart/form-data")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            return response.status, await response.json()


# API server will give the following responses:
# 1. If image upload is successful:
# Response Status: 200
# Response Payload:
#   {'Status': 'Success', 'filename': '104_2019-06-26T225415_00000_ANPR_L00.jpg'}
# 2. If there is a problem with upload:
# Response Status: 400
# Response Payload:
# {'Error': 'No file part'}
# Following code snippet shows how to use the post_data function above.
# In order to upload multiple images, call the post_data function in a loop.
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    img_dir = "/home/srikar/mobilenet/test/"
    files = os.listdir(img_dir) 
    for i in files:
        fname = os.path.join(img_dir, i)
        image_file = open(fname, 'rb')
        image_fname = i
        response_status, response_payload = loop.run_until_complete(post_data(
            image=image_file,
            image_filename=image_fname,
        ))

        print("Response status: ", response_status)
        print("Response reply: ", response_payload)
