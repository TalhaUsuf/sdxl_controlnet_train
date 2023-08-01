import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_images_side_by_side(image_path1, image_path2):
    # Read the images using matplotlib.image.imread
    img1 = mpimg.imread(image_path1)
    img2 = mpimg.imread(image_path2)

    # Create a figure and axis using matplotlib.pyplot.subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the images on the corresponding axes
    axes[0].imshow(img1)
    axes[0].axis('off')  # Turn off the axis for better visualization
    axes[0].set_title('Image 1')

    axes[1].imshow(img2)
    axes[1].axis('off')  # Turn off the axis for better visualization
    axes[1].set_title('Image 2')
    plt.savefig("combined.png", dpi=400)
    # Show the plot
    # plt.show()

# Example usage:
# Replace 'path_to_image1.jpg' and 'path_to_image2.jpg' with the actual image paths
plot_images_side_by_side('original.png', 'output.png')
