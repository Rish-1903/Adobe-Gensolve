import cv2
import numpy as np
import os
import random



def generate_circle(image, center, radius, color):
    num_points = 20
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        random_offset = random.randint(-5, 5)
        x = int(center[0] + (radius + random_offset) * np.cos(angle))
        y = int(center[1] + (radius + random_offset) * np.sin(angle))
        points.append((x, y))
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    for _ in range(5):
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        offset_points = points + [offset_x, offset_y]
        cv2.polylines(image, [offset_points], isClosed=True, color=color, thickness=random.randint(1, 3), lineType=cv2.LINE_AA)

def generate_line(image, start, end, color):
    num_segments = 5
    max_offset = 2

    points = [start]
    for i in range(1, num_segments):
        t = i / num_segments
        intermediate_point = (
            int(start[0] * (1 - t) + end[0] * t + random.randint(-max_offset, max_offset)),
            int(start[1] * (1 - t) + end[1] * t + random.randint(-max_offset, max_offset))
        )
        points.append(intermediate_point)
    points.append(end)

    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i + 1], color, thickness=random.randint(1, 3), lineType=cv2.LINE_AA)

def generate_rectangle(image, top_left, width, height, color):
    variability = lambda: random.randint(-10, 10)

    top_left = (top_left[0] + variability(), top_left[1] + variability())
    top_right = (top_left[0] + width + variability(), top_left[1] + variability())
    bottom_left = (top_left[0] + variability(), top_left[1] + height + variability())
    bottom_right = (top_left[0] + width + variability(), top_left[1] + height + variability())

    generate_line(image, top_left, top_right, color)
    generate_line(image, top_right, bottom_right, color)
    generate_line(image, bottom_right, bottom_left, color)
    generate_line(image, bottom_left, top_left, color)

def generate_star(image, center, size, color):
    points = []
    for i in range(10):
        angle = i * np.pi / 5
        length = size if i % 2 == 0 else size / 2
        length += random.randint(-10, 10)
        x = int(center[0] + length * np.cos(angle))
        y = int(center[1] - length * np.sin(angle))
        points.append((x, y))

    for j in range(10):
        start = points[j]
        end = points[(j + 1) % 10]
        generate_line(image, start, end, color)

def generate_ellipse(image, center, axes, angle, color):
    num_points = 10
    points = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        random_offset = random.randint(-5, 5)
        x = int(center[0] + (axes[0] + random_offset) * np.cos(theta) * np.cos(angle) - (axes[1] + random_offset) * np.sin(theta) * np.sin(angle))
        y = int(center[1] + (axes[0] + random_offset) * np.cos(theta) * np.sin(angle) + (axes[1] + random_offset) * np.sin(theta) * np.cos(angle))
        points.append((x, y))
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    for _ in range(5):
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        offset_points = points + [offset_x, offset_y]
        cv2.polylines(image, [offset_points], isClosed=True, color=color, thickness=random.randint(1, 3), lineType=cv2.LINE_AA)

def generate_polygon(image, center, sides, radius, color):
    points = []
    for i in range(sides):
        angle = i * 2 * np.pi / sides
        variable_radius = radius + random.randint(-10, 10)
        x = int(center[0] + variable_radius * np.cos(angle))
        y = int(center[1] - variable_radius * np.sin(angle))
        points.append((x, y))

    for i in range(sides):
        start = points[i]
        end = points[(i + 1) % sides]
        generate_line(image, start, end, color)

shapes = ['circle', 'rectangle', 'star',  'polygon', 'ellipse', 'line']
output_dir = './dataset'
os.makedirs(output_dir, exist_ok=True)

# Create train and valid directories with subdirectories for each shape
train_dir = os.path.join(output_dir, 'train')
valid_dir = os.path.join(output_dir, 'valid')

for shape in shapes:
    os.makedirs(os.path.join(train_dir, shape), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, shape), exist_ok=True)

def generate_images(shape_type, file_path,directory):
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    color = (0, 0, 0)  # Black color for shapes

    shape_drawers = {
        'circle': lambda: generate_circle(img, (128, 128), 100, color),
        'rectangle': lambda: generate_rectangle(img, (50, 50), 150, 100, color),
        'star': lambda: generate_star(img, (128, 128), 100, color),
        'polygon': lambda: generate_polygon(img, (128, 128), 8, 100, color),
        'ellipse': lambda: generate_ellipse(img, (128, 128), (100, 50), 0, color),
        'line': lambda: generate_line(img, (50, 128), (200, 128), color)
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

print('Image generation completed.')

