from utils import * 
from dotenv import load_dotenv


# Main function
def main(image_path, n_clusters=5):
    # Load and process image
    original_img = load_image(image_path)
    img_data = reshape_image(original_img)

    # Plot original image distribution
    plot_3d_scatter(img_data, img_data, "Original Image Color Distribution")

    # Perform k-means
    kmeans = perform_kmeans(img_data, n_clusters)

    # Plot clustered image distribution
    plot_3d_scatter_res(img_data, kmeans.labels_, "Clustered Image Color Distribution")

    # Reconstruct segmented image
    segmented_img = reconstruct_image(kmeans.labels_, kmeans.cluster_centers_, original_img.shape)

    # Display original and segmented images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(original_img)
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2.imshow(segmented_img.astype(np.uint8))
    ax2.set_title("Segmented Image")
    ax2.axis('off')
    plt.show()

# Run the main function
if __name__ == "__main__":
    # image_path = "path/to/your/image.jpg"  # Replace with your image path
    # main(image_path, n_clusters=5)
    image_path = "assets\lena_color.png"
    # img = load_image(image_path)
    # image = reshape_image(img)
    # plot_3d_scatter(image, image, "Color Distribution")
    
    main(image_path)