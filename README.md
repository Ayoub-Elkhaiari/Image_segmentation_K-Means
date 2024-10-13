# Image Segmentation using K-Means Clustering

This project implements image segmentation using the K-Means clustering algorithm. It visualizes the color distribution of an image in 3D space and performs segmentation based on color similarity.

## Features

- Load and preprocess images
- Visualize color distribution in 3D space
- Perform K-Means clustering for image segmentation
- Display original and segmented images side by side

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Ayoub-Elkhaiari/Image_segmentation_K-Means.git
   cd Image_segmentation_K-Means
   ```

2. Install the required dependencies:
   ```
   pip install numpy matplotlib scikit-learn opencv-python 
   ```

## Usage

1. Place your image in the `assets` folder or update the `image_path` in `main.py`.

2. Run the main script:
   ```
   python main.py
   ```

3. The script will display:
   - 3D scatter plot of the original image color distribution
   - 3D scatter plot of the clustered image color distribution
   - Side-by-side comparison of the original and segmented images

## Code Structure

- `utils.py`: Contains utility functions for image processing, K-Means clustering, and visualization.
- `main.py`: The main script that orchestrates the image segmentation process.

### Key Functions Explained

1. `load_image(file_path)`:
   - Loads an image from the given file path using OpenCV.
   - Converts the image from BGR to RGB color space.
   - This function is crucial for preparing the image data for processing.

2. `reshape_image(img)`:
   - Reshapes the 3D image array (height x width x 3) into a 2D array (pixel_count x 3).
   - This transformation is necessary for applying K-Means clustering to the image data.

3. `perform_kmeans(data, n_clusters)`:
   - Applies K-Means clustering to the reshaped image data.
   - The `n_clusters` parameter determines the number of color segments in the output.
   - Returns the fitted KMeans model, which contains cluster assignments and centroids.

4. `plot_3d_scatter(data, labels, title, cluster_centers=None)`:
   - Creates a 3D scatter plot of the image pixels in RGB color space.
   - Useful for visualizing the color distribution of the original image.

5. `plot_3d_scatter_res(data, labels, title, cluster_centers=None)`:
   - Similar to `plot_3d_scatter`, but colors the points based on their cluster assignments.
   - Optionally plots cluster centers, providing a visual representation of the segmentation.

6. `reconstruct_image(labels, centers, shape)`:
   - Reconstructs the segmented image using cluster assignments and centroids.
   - Each pixel is replaced with its corresponding cluster center color.

7. `main(image_path, n_clusters=5)`:
   - Orchestrates the entire image segmentation process.
   - Loads the image, performs clustering, creates visualizations, and displays results.

These functions work together to load an image, transform its data, apply K-Means clustering for segmentation, visualize the results, and reconstruct the segmented image.

## Customization

You can adjust the number of clusters by changing the `n_clusters` parameter in the `main()` function call:

```python
main(image_path, n_clusters=7)  # Change to desired number of clusters
```

## Results

for K = 2:
![Screenshot 2024-10-13 174306](https://github.com/user-attachments/assets/e4e83704-5354-434c-ae3b-c3e032f2931e)

![Screenshot 2024-10-13 174401](https://github.com/user-attachments/assets/c955a14e-42f0-4d30-af18-333394c06d5a)

![Screenshot 2024-10-13 174412](https://github.com/user-attachments/assets/982399c8-2275-4372-924f-4f441185e537)

## Dependencies

- numpy
- matplotlib
- scikit-learn
- opencv-python




