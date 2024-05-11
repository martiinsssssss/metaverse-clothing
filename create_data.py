import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from io import BytesIO
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

def scrape_artist_images(song_name, artist_name, num_images=5):
    # Format the artist name for the URL
    formatted_artist_name = artist_name.replace(' ', '+')
    formatted_song_name = song_name.replace(' ', '+')
    formatted_artist_name = formatted_song_name+'+'+formatted_artist_name
    #print(formatted_artist_name)
    # URL to search for images of the artist
    url = f'https://www.google.com/search?q={formatted_artist_name}&tbm=isch'
    # Send HTTP GET request
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all image tags
        img_tags = soup.find_all('img')
        # Extract image URLs
        image_urls = [img['src'] for img in img_tags]
        # Download and return the images
        images = []
        for url in image_urls:
            try:
                # Send HTTP GET request to download the image
                img_response = requests.get(url)
                
                # Open the image using PIL
                img = Image.open(BytesIO(img_response.content))

                # Append the image to the list
                images.append(img)

                # Stop if the desired number of images is reached
                if len(images) == num_images:
                        break
            except Exception:
                pass
        return images
    else:
        print('Failed to retrieve images.')
        return []

def extract_colors(images, num_colors=10):

    pixels = []
    for image in images:
        # Convert the image from BGR to RGB (OpenCV reads images in BGR format)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape the image to a 2D array of pixels
        img_pixels = image.reshape((-1, 3))
        
        # add the pixels to the list
        pixels.extend(img_pixels)

    # remove pure white pixels
    pixels = [pixel for pixel in pixels if not all([value == 255 for value in pixel])]

    # remove pure black pixels
    pixels = [pixel for pixel in pixels if not all([value == 0 for value in pixel])]

    # remove pixels that are almost black
    pixels = [pixel for pixel in pixels if not all([value < 10 for value in pixel])]

    # Initialize K-means clustering algorithm
    kmeans = KMeans(n_clusters=num_colors)
    
    # Fit K-means to the pixel data
    kmeans.fit(pixels)
    
    # Get the cluster centers (representative colors)
    colors = kmeans.cluster_centers_  

    # sort the colors taking into account the counts and color saturation, making a balance between the two.
    saturation = np.max(colors, axis=1) - np.min(colors, axis=1)
    sorted_indices = np.argsort(saturation)[::-1]
    colors = colors[sorted_indices].astype(int)

    #colors = [color for color in colors if not all([value < 20 for value in color])]
    
    return colors

# Spotify data
spotify = pd.read_csv('spotify.csv')
print(len(spotify))
spotify.head()
# Get popular songs
popular_songs = spotify[spotify['popularity'] > 50]

# get the artist name and song name
artist_name = popular_songs['artists'].values
song_name = popular_songs['track_name'].values

# put them in a list
artist_song = list(zip(song_name, artist_name))
artist_song = list(set(artist_song))

# get the images
for song_name, artist_name in tqdm(artist_song):
    images = scrape_artist_images(song_name, artist_name, num_images=5)
    
    if images!=[]:
        
        # image is <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=162x91 at 0x29FC571C0>
        # convert it to cv2 image
        images = [np.array(img) for img in images]

        # Extract colors from the images

        colors = extract_colors(images, num_colors=10)

        # add the colors to the dataframe
        colors = [tuple(color) for color in colors]
        #colors = list(set(colors))
        colors = [str(color) for color in colors]
        colors = '; '.join(colors)
        popular_songs.loc[(popular_songs['track_name'] == song_name) & ( popular_songs['artists'] == artist_name), 'colors'] = colors
        
#save the dataframe
popular_songs.to_csv('data.csv', index=False)
print('Data saved successfully.')