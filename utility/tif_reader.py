from utility.py_import import np, dataclass, os, cv2
import tifffile
from matplotlib.pyplot import imsave


@dataclass
class tifReader:
    img_path: str

    def __post_init__(self):
        self.__base_name = os.path.splitext(
            os.path.basename(self.img_path)
            )[0]
        
        self.length = 0
        self.__tif_images = []
        self.__read_images()


    def __read_images(self) -> None:
        """
        Read all images from the TIFF file and store them as grayscale images.
        """   
        try:
            tiff_data = tifffile.imread(self.img_path)
            self.__tif_images = [
                cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for img in tiff_data
                ]
            self.length = len(self.__tif_images)

        except IOError:
            raise Exception("Could not open or read the image file.")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")
        
        return
        

    def average(self, save_path:str=None) -> np.ndarray:
        """
        Obtain an average image of all images in the TIFF file.
        """
        if self.length <= 0:
            raise Exception("No images found.")

        images = [img_tuple for img_tuple in self.__tif_images]
        averaged_img = np.mean(images, axis=0)

        if save_path:
            filename = f"{self.__base_name}.png"
            imsave(os.path.join(save_path, filename), averaged_img.astype(np.float32))

        return averaged_img


    def average_indicies(self, index1:int, index2:int, save_path:str=None):
        """
        Obtain an average image of selected indicies
        """
        if self.length <= 0:
            raise Exception("No images found.")

        # Validate indices
        if index1 < 0 or index2 < 0 or index1 >= self.length or index2 >= self.length:
            raise Exception(f"Indices out of range. Valid range: 0 to {self.length - 1}")

        initial = min(index1, index2)
        end = max(index1, index2) + 1
        averaged_img = np.mean(self.__tif_images[initial:end], axis=0)

        if save_path:
            filename = f"{self.__base_name}.png"
            imsave(os.path.join(save_path, filename), averaged_img.astype(np.float32))

        return averaged_img


    def get_image(self, index: int):
        """
        Get the image at the specified index
        """
    
        if index < 0 or index >= self.length:
            raise Exception(f"Out of Bound Error, {index}. Valid range: 0 to {self.length - 1}")
        return self.__tif_images[index]