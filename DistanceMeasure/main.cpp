/*
 * OpenCV + ARToolKit + OpenGL = Augmented Reality Megacombo :)
 *
 * This example uses fiducial markers to calculate their distance from the (calibrated)
 * camera. Also, displays the obligatory Utah Teapot on the marker ;)
 *
 * For this example to work, you need:
 *
 * a) OpenCV (http://opencv.willowgarage.com/wiki/)
 * b) ARToolKit from Launchpad (https://launchpad.net/artoolkitplus)
 * c) OpenGL obviously
 *
 * Copyright: Michal Kottman 2010, MIT license (use your Google-fu to look it up)
 *
 *      Author: Michal Kottman
 */

#include <stdlib.h>
#include <stdio.h>

#include <ARToolKitPlus/ARToolKitPlus.h>
#include <ARToolKitPlus/Tracker.h>
#include <ARToolKitPlus/TrackerSingleMarker.h>
#include <ARToolKitPlus/TrackerMultiMarker.h>
#include <ARToolKitPlus/Camera.h>

#include <opencv2/opencv.hpp>

#include <GL/freeglut.h>

#include <vector>

// lets hope there are no conflicts ;)
using namespace ARToolKitPlus;
using namespace cv;
using namespace std;

#define WIDTH       640
#define HEIGHT      480
#define PLANE_NEAR  1.f
#define PLANE_FAR   1000.f

/**
 * A class that enables conversion from OpenCV calibration matrices. Derives from Camera
 * in order to access the (protected) internals of Camera class.
 */
class OpenCVCamera : Camera {
public:
    /**
     * Takes the OpenCV camera matrix and distortion coefficients, and generates
     * ARToolKitPlus compatible Camera.
     */
    static Camera * fromOpenCV(const Mat& cameraMatrix, const Mat& distCoeffs) {
        OpenCVCamera *cam = new OpenCVCamera;

        cam->xsize = WIDTH;
        cam->ysize = HEIGHT;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                cam->mat[i][j] = 0;

        float fx = cam->fc[0] = (float) cameraMatrix.at<double> (0, 0);
        float fy = cam->fc[1] = (float) cameraMatrix.at<double> (1, 1);
        float cx = cam->cc[0] = (float) cameraMatrix.at<double> (0, 2);
        float cy = cam->cc[1] = (float) cameraMatrix.at<double> (1, 2);

        cam->mat[0][0] = fx;
        cam->mat[1][1] = fy;
        cam->mat[0][2] = cx;
        cam->mat[1][2] = cy;
        cam->mat[2][2] = 1.0;

        for (int i = 0; i < 4; i++) {
            float f = (float) distCoeffs.at<double> (i);
            cam->kc[i] = f;
        }
        cam->kc[4] = 0;
        cam->kc[5] = 0;
        cam->undist_iterations = 1;

        return cam;
    }
};

/** Prints out camera matrix to stdout */
void dumpCamera(Camera *cam) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.4f ", cam->mat[i][j]);
        }
        printf("\n");
    }
}

/** Prints out a Mat to stdout with optional name */
void dumpMatrix(const Mat &mat, const char *name = "matrix") {
    const int t = mat.type();
    printf("==== %s ====\n", name);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            switch (t) {
            case CV_32F:
                printf("%6.4f ", mat.at<float> (i, j));
                break;
            case CV_64F:
                printf("%6.4f ", mat.at<double> (i, j));
                break;
            }
        }
        printf("\n");
    }
    printf("====----====\n");
}

/*
 * Common globals
 */

GLuint gCameraTextureId;
const ARFloat *gModelViewMatrix;
const ARFloat *gProjectionMatrix;
Mat gCameraImage, gResultImage;
Camera *gCamera;
double gDistance;
float gLightPos[4] = { 0, 5, 0, 1 };
int gFinished;

/*
 * GLUT callbacks
 */

/**
 * OpenGL initialization
 */
void glInit() {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gCameraTextureId);

    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
}

/**
 * Updates texture handle gCameraTextureId with OpenCV image in cv::Mat from gResultImage
 */
void updateTexture() {
    glBindTexture(GL_TEXTURE_2D, gCameraTextureId);

    // set texture filter to linear - we do not build mipmaps for speed
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // create the texture from OpenCV image data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_BGR,
            GL_UNSIGNED_BYTE, gResultImage.data);
}

/**
 * Draw the background from the camera image
 */
void drawBackground() {
    // set up the modelview matrix so that the view is between [-1,-1] and [1,1]
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // draw the quad textured with the camera image
    glBindTexture(GL_TEXTURE_2D, gCameraTextureId);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1);
    glVertex2f(-1, -1);
    glTexCoord2f(0, 0);
    glVertex2f(-1, 1);
    glTexCoord2f(1, 0);
    glVertex2f(1, 1);
    glTexCoord2f(1, 1);
    glVertex2f(1, -1);
    glEnd();

    // reset the projection matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

/** OpenGL display callback */
void displayFunc() {
    glClear(GL_COLOR_BUFFER_BIT);

    // render the background image from camera texture
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    drawBackground();

    // clear th depth buffer bit so that the background is overdrawn
    glClear(GL_DEPTH_BUFFER_BIT);

    // everything will be white
    glColor3f(1, 1, 1);

    // start with fresh modelview matrix and apply the transform of the marker
    glLoadIdentity();
    glMultMatrixf(gModelViewMatrix);

    char distStr[32];
    sprintf(distStr, "Distance: %8.4lf", gDistance);
    glutSetWindowTitle(distStr);

    // some rotations so that the scene is not static
    static float angle = 0.0;
    glRotatef(90, 1, 0, 0);
    glRotatef(angle += 1, 0, 1, 0);

    // render a lighted teapot from above
    // enable the texture for a nice effect ;)
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glLightfv(GL_LIGHT0, GL_POSITION, gLightPos);
    glTranslatef(0, 0.5, 0);
    glutSolidTeapot(0.5);

    glutSwapBuffers();
    glutPostRedisplay();
}

/** Windows resize callback */
void reshape(GLint width, GLint height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMultMatrixf(gProjectionMatrix);
    glMatrixMode(GL_MODELVIEW);
}

/** Keyboard callback */
void keyFunc(unsigned char key, int x, int y) {
    switch (key) {
    case 27:
        gFinished = true;
        break;
    case 'c': {
        dumpCamera(gCamera);
        break;
    }
    case 'p': {
        GLfloat mat[4 * 4];
        glGetFloatv(GL_PROJECTION_MATRIX, mat);
        Mat projection(4, 4, CV_32FC1, mat);
        dumpMatrix(projection, "projection");
        break;
    }
    case 'm': {
        Mat modelView(4, 4, CV_32FC1, (void*) gModelViewMatrix);
        dumpMatrix(modelView, "modelview");
        break;
    }
    }
}

/*
 * Support functions
 */

/**
 * Calculates the distance from eye/camera to the marker.
 */
void updateDistanceFromMarker() {
    // Get the modelview matrix
    Mat modelView = Mat(4, 4, CV_32FC1, (void*) gModelViewMatrix);

    // Translation is stored in m41, m42, m43
    Mat translation = modelView(Rect(0, 3, 3, 1));

    // The distance is measured in "marker sizes", so multiply it by marker size
    // in my case, it 8cm big
    gDistance = norm(translation) * 8;
}


/*
 * Main program`
 */

int main(int argc, char *argv[]) {
#if 0
    // capture from file
    VideoCapture cap("camera.avi");
#else
    // capture from camera
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);
    if (!cap.isOpened()) {
        cerr << "Failed to open camera!\n";
        return -1;
    }
#endif

    // load the camera calibration data stored in calibration.xml
    Mat camMat, distCoeff;
    FileStorage fs("calibration.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open calibration.xml (did you calibrate the camera?)\n";
        return -1;
    }
    fs["calib"] >> camMat;
    fs["dist"] >> distCoeff;
    gCamera = OpenCVCamera::fromOpenCV(camMat, distCoeff);

    // setup the ARToolKitPlus tracker
    TrackerSingleMarker tracker(WIDTH, HEIGHT);

    // When you supply OpenGL near and far planes, it calculates the projection matrix
    tracker.setCamera(gCamera, PLANE_NEAR, PLANE_FAR);
    // Do not load from file - it is already supplied
    tracker.init(NULL, 0, 0);
    // Take the pattern to be size of 8cm - each "pixel" is 1cm
    tracker.setPatternWidth(1);
    // Black border width for BCH markers is 1/8 of the marker side
    tracker.setBorderWidth(0.125);
    // In varying light conditions, try
    tracker.setNumAutoThresholdRetries(3);
    // otherwise, try
    // tracker.setThreshold(128);

    // OpenCV loads and stores images in BGR order
    tracker.setPixelFormat(PIXEL_FORMAT_BGR);
    // Use look-up tables for undistortion - fast, but not for high-res images
    tracker.setUndistortionMode(UNDIST_LUT);
    // I use BCH markers, other possible modes: SIMPLE and TEMPLATE
    tracker.setMarkerMode(MARKER_ID_BCH);
    // Use the Robust Pose Estimator for pose tracking
    tracker.setPoseEstimator(POSE_ESTIMATOR_RPP);

    gProjectionMatrix = tracker.getProjectionMatrix();
    ARFloat dummyMatrix[16] = {0};
    gModelViewMatrix = dummyMatrix;

    // Setup GLUT rendering
    glutInit(&argc, argv);
    glutCreateWindow("Main");
    glutKeyboardFunc(keyFunc);
    glutReshapeFunc(reshape);
    glutReshapeWindow(WIDTH, HEIGHT);
    glutDisplayFunc(displayFunc);

    glInit();

    gFinished = false;
    while (!gFinished) {
        cap >> gCameraImage;

        // find the markers on the image
        ARMarkerInfo *markers;
        int nMarkers;
        vector<int> ids = tracker.calc(gCameraImage.data, &markers,
                &nMarkers);

#if 0
        // this is very slow compared to the rest of the program and ...
        undistort(gCameraImage, gResultImage, camMat, distCoeff);
#else
        // ... no undistortion is needed with a relatively good camera
        gResultImage = gCameraImage;
#endif

        // draw the marker corners and IDs using OpenCV
        for (int i = 0; i < nMarkers; i++) {
            ARMarkerInfo &m = markers[i];
            Point center(m.pos[0], m.pos[1]);
            for (int j = 0; j < 4; j++) {
                Point p(m.vertex[j][0], m.vertex[j][1]);
                circle(gResultImage, p, 6, Scalar(255, 0, 255));
            }
            char txt[20];
            sprintf(txt, "%d", m.id);
            putText(gResultImage, txt, center, CV_FONT_HERSHEY_SIMPLEX, 1,
                    Scalar(0, 255, 0));
        }

        // no need to show the image - it will be displayed in OpenGL
        // imshow("display", gResultImage);

        // this is needed to get the modelView matrix
        if (nMarkers > 0) {
            // either select a concrete marker by it's ID:
            // tracker.selectDetectedMarker(0);
            // or find the best marker available
            tracker.selectBestMarkerByCf();
            gModelViewMatrix = tracker.getModelViewMatrix();
        }

        updateTexture();
        updateDistanceFromMarker();

        glutMainLoopEvent();
    }

    return 0;
}
