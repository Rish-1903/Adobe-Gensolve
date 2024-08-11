# CurveTopia
## Task1 Regularizing Curve
Curvetopia is a project focused on regularizing hand-drawn curves by identifying and standardizing regular shapes within them. The algorithm targets the following shapes:

- Straight Lines
- Circles and Ellipses
- Rectangles and Rounded Rectangles
- Regular Polygons
- Star Shapes

The algorithm can optionally convert line art images into polylines, and it will be tested on various images to ensure its effectiveness in distinguishing and regularizing these shapes.

To run the above tasks in a folder:

1. Run the script `create_dataset.py`.
3. Then run the script `model_training.py` for model training and evaluation. Change the path of the image to be predicted.

## Task 2 Exploring Symmetries in Curves

To identify reflection symmetries in various closed shapes, transform the shape into a set of points and follow these steps:

1. **Straight Line:**
   - Check for any line perpendicular to the given line at its midpoint.

2. **Circle:**
   - Identify any diameter as a line of symmetry.

3. **Ellipse:**
   - Identify the major and minor axes as lines of symmetry.

4. **Polygon:**
   - **Regular Polygon:** Count the number of sides to find the number of symmetry lines.
   - **Irregular Polygon:** Manually check for lines that divide the polygon into mirrored halves.

5. **Star:**
   - Identify lines passing through vertices and opposite indents as lines of symmetry.

6. **Rectangle:**
   - Identify lines passing through the midpoints of opposite sides (horizontal and vertical).
     
To run the above task :

1.Run the script `symmetry.py` by inputting the path to your csv file and horizontal/vertical symmetry.

## Task3 CompletingIncomplete Curves

Run the script  `unmasking.py` by inputting the image path in .png or .jpg format.




