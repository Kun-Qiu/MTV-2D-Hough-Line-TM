import numpy as np
import os
from PIL import Image
from matplotlib.pyplot import imsave


class tifReader:

    def __init__(self, img_path):
        """
        Default Constructor
        :param img_path:         The tif image path or file object
        """
        self.__img = img_path
        self.__base_name = os.path.basename(img_path)
        self.__base_name = os.path.splitext(self.__base_name)[0]
        
        self.__img_uint8 = None 
        self.__array_length = 0
        self.__tif_images = []
        self.__read_images()


    def average_all_tif(self, save_path=None):
        """
        Obtain an average image of the input images
        
        :param save_path: Path to which the image will be saved at
        :return: The averaged of all image
        """

        if self.__array_length <= 0:
            raise Exception("No images found.")

        images = [img_tuple for img_tuple in self.__tif_images]
        averaged_img = np.mean(images, axis=0)

        if save_path:
            filename = f"{self.__base_name}.png"
            imsave(os.path.join(save_path, filename), averaged_img.astype(np.float32))

        return averaged_img


    def average_tif(self, index1, index2, save_path=None):
        """
        From multiple images obtain an average image of the input images
        :param save_path:       Path to which the image will be saved at
        :param index1:          First image of the series to be averaged
        :param index2:          Second image of the series to be averaged
        :return: The averaged image of the tif
        """

        if self.__array_length <= 0:
            raise Exception("No images found.")

        # Validate indices
        if index1 < 0 or index2 < 0 or index1 >= self.__array_length or index2 >= self.__array_length:
            raise Exception(f"Indices out of range. Valid range: 0 to {self.__array_length - 1}")

        initial = min(index1, index2)
        end = max(index1, index2) + 1
        averaged_img = np.mean(self.__tif_images[initial:end], axis=0)

        if save_path:
            filename = f"{self.__base_name}.png"
            imsave(os.path.join(save_path, filename), averaged_img.astype(np.float32))

        return averaged_img


    def save_tiff(self, index1, index2, save_path):
        if self.__array_length <= 0:
            raise Exception("No images found.")

        # Validate indices
        if index1 < 0 or index2 < 0 or index1 >= self.__array_length or index2 >= self.__array_length:
            raise Exception(f"Indices out of range. Valid range: 0 to {self.__array_length - 1}")

        initial = min(index1, index2)
        end = max(index1, index2) + 1
        for i in range (initial, end):
            filename = f"{self.__base_name}_{i:04d}.png"
            imsave(os.path.join(save_path, filename), self.__tif_images[i].astype(np.float32))


    def get_image(self, index):
        """
        Get the image at the specified index
        :param index: The index of the image
        :return: The image at that index
        """

        if index < 0 or index >= self.__array_length:
            raise Exception(f"Out of Bound Error, {index}. Valid range: 0 to {self.__array_length - 1}")
        return self.__tif_images[index]


    def __read_images(self):
        """
        Access the image in tif and store in an array of images
        :return: None
        """

        try:
            img = Image.open(self.__img)

            # Ensure it's a tif file
            if img.format != 'TIFF':
                raise Exception("File is not a TIFF image.")

            self.__tif_images.clear()  # Clear previous data if any
            self.__array_length = 0  # Reset array length

            for i in range(img.n_frames):
                img.seek(i)

                # Convert to NumPy array
                frame_array = np.array(img)
                self.__tif_images.append(frame_array)
                self.__array_length += 1

        except IOError:
            raise Exception("Could not open or read the image file.")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")


    def __convert2uint8(self, img):
        """
        Convert a np.array consisting of float32 or 64 to uint8

        :param img: input image to be converted
        :return: None
        """
        try:
            imin = img.min()
            imax = img.max()

            # Handle case where all values are the same (avoid division by zero)
            if imax == imin:
                new_img = np.zeros_like(img, dtype=np.uint8)
            else:
                # Normalize and convert to uint8
                copy_img = img.astype(np.float32)
                a = 255.0 / (imax - imin)
                b = -a * imin
                new_img = (a * copy_img + b).astype(np.uint8)
            
            self.__img_uint8 = new_img

        except Exception as e:
            raise Exception("Failed to convert image to uint8. Original error: {}".format(e))


    def get_uint8(self, img):
        """
        Get the uint8 image from the object
        
        :param img: The input image
        :return: uint8 image
        """
        self.__convert2uint8(img)
        return self.__img_uint8


    def get_tif_size(self):
        """
        Get the number of images in the TIFF file
        :return: Number of images
        """
        return self.__array_length