import streamlit as st
import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import createdataset
import modeltraining
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
# Title of the application
st.title("Curvetopia Image Processing")

# Sidebar for navigation
app_mode = st.sidebar.selectbox("Choose the operation", ["Generate Dataset", "Train Model", "Evaluate Model", "Symmetry Detection", "Unmasking"])

# Directory paths
output_dir = './dataset'
train_dir = os.path.join(output_dir, 'train')
valid_dir = os.path.join(output_dir, 'valid')

# Function to generate dataset
def generate_dataset():
    shapes = ['circle', 'rectangle', 'star',  'polygon', 'ellipse', 'line']
    os.makedirs(output_dir, exist_ok=True)
    for shape in shapes:
        os.makedirs(os.path.join(train_dir, shape), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, shape), exist_ok=True)
    
    def generate_images(shape_type, file_path, directory):
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        color = (0, 0, 0)  # Black color for shapes

        shape_drawers = {
            'circle': lambda: createdataset.generate_circle(img, (128, 128), 100, color),
            'rectangle': lambda: createdataset.generate_rectangle(img, (50, 50), 150, 100, color),
            'star': lambda: createdataset.generate_star(img, (128, 128), 100, color),
            'polygon': lambda: createdataset.generate_polygon(img, (128, 128), 8, 100, color),
            'ellipse': lambda: createdataset.generate_ellipse(img, (128, 128), (100, 50), 0, color),
            'line': lambda: createdataset.generate_line(img, (50, 128), (200, 128), color)
        }

        if shape_type in shape_drawers:
            shape_drawers[shape_type]()

        img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite(os.path.join(directory, file_path), img)

    for shape in shapes:
        for i in range(400):
            file_name = f'{shape}_{i}.png'
            if i < 320:  # 80% for training
                generate_images(shape, file_name, os.path.join(train_dir, shape))
            else:  # 20% for validation
                generate_images(shape, file_name, os.path.join(valid_dir, shape))

    st.write('Image generation completed.')

# Function to train model
def train_model():
    batch_size = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images if necessary
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root='dataset/train/', transform=transform)
    test_dataset = datasets.ImageFolder(root='dataset/valid', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = modeltraining.ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    n_total_steps = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 2000 == 0:
                st.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model.pth')
    st.write('Finished Training')

# Function for symmetry detection
def symmetry_detection(csv_path, axis, output_csv_path):
    def read_csv(csv_path):
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
        path_XYs = []

        for i in np.unique(np_path_XYs[:, 0]):
            npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
            XYs = []

            for j in np.unique(npXYs[:, 0]):
                XY = npXYs[npXYs[:, 0] == j][:, 1:]
                XYs.append(XY)

            path_XYs.append(XYs)

        return path_XYs

    def reflect_points(points, axis, center):
        reflected_points = np.copy(points)
        if axis == 'vertical':
            reflected_points[:, 0] = 2 * center[0] - points[:, 0]
        elif axis == 'horizontal':
            reflected_points[:, 1] = 2 * center[1] - points[:, 1]
        else:
            raise ValueError("Axis must be 'vertical' or 'horizontal'")
        return reflected_points

    def is_symmetric(points, axis):
        center = np.mean(points, axis=0)
        reflected_points = reflect_points(points, axis, center)
        return np.allclose(points, reflected_points, atol=1e-2)

    def plot_points_with_symmetry(points, axis, is_symmetric):
        plt.figure()
        plt.scatter(points[:, 0], points[:, 1], c='b', label='Original Points')
        center = np.mean(points, axis=0)
        if axis == 'vertical':
            plt.axvline(x=center[0], color='g' if is_symmetric else 'r', linestyle='--', label='Symmetry Line')
        elif axis == 'horizontal':
            plt.axhline(y=center[1], color='g' if is_symmetric else 'r', linestyle='--', label='Symmetry Line')
        plt.title('Symmetry Detection')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        st.pyplot(plt)

    def save_results_to_csv(points, is_symmetric, output_path):
        df = pd.DataFrame(points, columns=['x', 'y'])
        df['is_symmetric'] = is_symmetric
        df.to_csv(output_path, index=False)

    path_XYs = read_csv(csv_path)
    data_frames = []
    for path in path_XYs:
        for segment in path:
            df = pd.DataFrame(segment, columns=['x', 'y'])
            data_frames.append(df)
    final_df = pd.concat(data_frames, ignore_index=True)
    x = final_df['x'].values
    y = final_df['y'].values
    all_points = np.column_stack((x, y))
    symmetric = is_symmetric(all_points, axis)
    plot_points_with_symmetry(all_points, axis, symmetric)
    save_results_to_csv(all_points, symmetric, output_csv_path)
    st.write(f'Symmetry along {axis}: {symmetric}')

# Function for unmasking
def unmasking(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [max_contour], -1, 255, -1)
    inverted_mask = cv2.bitwise_not(mask)
    completed_image = cv2.bitwise_and(image, image, mask=inverted_mask)
    st.image(completed_image, caption="Unmasked Image", use_column_width=True)

# Streamlit app logic
if app_mode == "Generate Dataset":
    st.subheader("Generate Dataset")
    if st.button("Generate"):
        generate_dataset()

elif app_mode == "Train Model":
    st.subheader("Train Model")
    if st.button("Train"):
        train_model()

elif app_mode == "Evaluate Model":
    st.subheader("Evaluate Model")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        transform = transforms.Compose([
                    transforms.Resize((128, 128)),  # Resize images if necessary
                    transforms.ToTensor()
                                        ])
        model = modeltraining.ConvNet()
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        predicted_class = modeltraining.predict_image(uploaded_file, model, transform, 'cpu')
        st.write(f'Predicted Class: {predicted_class}')

elif app_mode == "Symmetry Detection":
    st.subheader("Symmetry Detection")
    csv_path = st.text_input("Enter CSV path")
    axis = st.selectbox("Choose axis of symmetry", ["vertical", "horizontal"])
    if st.button("Check Symmetry"):
        symmetry_detection(csv_path, axis, 'symmetry_results.csv')

elif app_mode == "Unmasking":
    st.subheader("Unmasking")
    image_path = st.text_input("Enter image path")
    if st.button("Unmask"):
        unmasking(image_path)
