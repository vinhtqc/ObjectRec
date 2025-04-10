package com.example.objectrec;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Display;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.view.TextureView;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.objectrec.R;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import androidx.camera.view.PreviewView;
import androidx.camera.core.CameraInfo;

public class MainActivity extends AppCompatActivity {

    final private int MYDEBUG=0;

    final private int DROP_FRAME_COUNT = 5;

    private PreviewView previewView;

    private static final String TAG = "MainActivity";
    private static final int REQUEST_CAMERA_PERMISSION = 1;
    private TextureView textureView;
    private ImageView imageView;
    private Interpreter tflite;
    private int imageSizeX;
    private int imageSizeY;

    private int numClasses = 80;
    private float confidenceThreshold = 0.5F;
    private float nmsIoUThreshold = 0.5F;
    private List<String> labels;
    private Handler handler;
    private HandlerThread handlerThread;
    private int imageCnt=0;
    private Bitmap gbitmap;
    private int cameraOrientation = Surface.ROTATION_0;
    private OrientationEventListener orientationEventListener;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        imageView = findViewById(R.id.imageView);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        } else {
            startCamera();
        }

        handlerThread = new HandlerThread("videoThread");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        imageSizeY = 16;
        imageSizeX = 16;
        try {

            tflite = new Interpreter(FileUtil.loadMappedFile(this, "yolov8n.tflite")); // Replace with your model file name
            labels = FileUtil.loadLabels(this, "labels.txt"); // Replace with your labels file name
            numClasses = tflite.getOutputTensor(0).shape()[1] - 4; // Adjust based on your model's output

            int[] inputShape = tflite.getInputTensor(0).shape();
            imageSizeY = inputShape[2];
            imageSizeX = inputShape[1];

            if (MYDEBUG==1) {
                Bitmap bm = BitmapFactory.decodeResource(getResources(), R.drawable.fruit);
                gbitmap = Bitmap.createScaledBitmap(bm, imageSizeX, imageSizeY, true);
                imageView.setImageBitmap(gbitmap);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {

                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    Preview preview = new Preview.Builder().build();
                    CameraSelector cameraSelector = new CameraSelector.Builder()
                            .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

                    CameraInfo cameraInfo = cameraProvider.getCameraInfo(cameraSelector);
                    int sensorOrientation = cameraInfo.getSensorRotationDegrees();
                    cameraOrientation = getDisplayRotation();

                    preview.setSurfaceProvider(previewView.getSurfaceProvider());

                    ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                            .setTargetResolution(new Size(imageSizeX, imageSizeY))
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build();



                    imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), imageProxy -> {
                        if (MYDEBUG==0)
                        {
                            // Rotate the detected bitmap based on the camera orientation
                            Matrix matrix = new Matrix();
                            cameraOrientation = getDisplayRotation();
                            matrix.postRotate(cameraOrientation+90);

                            Bitmap bitmap = toBitmap(imageProxy);
                            gbitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

                        }

                        if (gbitmap != null) {
                            handler.post(() -> runObjectDetection(gbitmap));
                        }
                        imageProxy.close();
                    });

                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

                } catch (Exception e) {
                    Log.e(TAG, "Use case binding failed", e);
                }

        }, ContextCompat.getMainExecutor(this));
    }

    private int getDisplayRotation() {
        WindowManager windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        Display display = windowManager.getDefaultDisplay();
        int rotation = display.getRotation();
        switch (rotation) {
            case Surface.ROTATION_0:
                return 0;
            case Surface.ROTATION_90:
                return 90;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_270:
                return 270;
            default:
                return 0;
        }
    }

    private void rotateImageView(int rotationDegrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(-rotationDegrees, (float) imageView.getWidth() / 2, (float) imageView.getHeight() / 2);
        imageView.setImageMatrix(matrix);
        imageView.invalidate();
    }

    private Bitmap toBitmap(ImageProxy image) {
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        android.graphics.YuvImage yuvImage = new android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, image.getWidth(), image.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    /*private void runObjectDetection(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, true);
        TensorImage inputImageBuffer = TensorImage.fromBitmap(resizedBitmap);

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(imageSizeY, imageSizeX, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0, 255))
                .build();

        TensorImage processedImageBuffer = imageProcessor.process(inputImageBuffer);

        TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 80, 4}, DataType.FLOAT32);
        TensorBuffer outputFeature1 = TensorBuffer.createFixedSize(new int[]{1, 80}, DataType.FLOAT32);
        TensorBuffer outputFeature2 = TensorBuffer.createFixedSize(new int[]{1, 80}, DataType.FLOAT32);


        tflite.run(processedImageBuffer.getBuffer(), new Object[]{outputFeature0.getBuffer().rewind(), outputFeature1.getBuffer().rewind(), outputFeature2.getBuffer().rewind()});

        float[] locations = outputFeature0.getFloatArray(); // Corrected line.
        float[] classes = outputFeature1.getFloatArray();
        float[] scores = outputFeature2.getFloatArray();


       // float[] locations = new float[]{0.1f, 0.2f, 0.8f, 0.9f, 0.5f, 0.6f, 0.7f, 0.8f
          //      , 0.9f, 1.0f, 1.1f, 1.2f, 0, 0, 0, 0};
        //float[] classes = new float[]{0, 1, 2, 3};
        //float[] scores = new float[]{0.6f, 0.2f, 0.3f, 0.1f};

        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.RED);
        paint.setStrokeWidth(5);

        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > 0.5f) { // Adjust threshold as needed
                int classIndex = (int) classes[i];
                String label = labels.get(classIndex);

                float ymin = locations[i * 4 + 0] * bitmap.getHeight();
                float xmin = locations[i * 4 + 1] * bitmap.getWidth();
                float ymax = locations[i * 4 + 2] * bitmap.getHeight();
                float xmax = locations[i * 4 + 3] * bitmap.getWidth();

                RectF rect = new RectF(xmin, ymin, xmax, ymax);
                canvas.drawRect(rect, paint);

                paint.setStyle(Paint.Style.FILL);
                paint.setColor(Color.WHITE);
                canvas.drawText(label, xmin, ymin - 10, paint);
                paint.setStyle(Paint.Style.STROKE);
            }
        }
        runOnUiThread(() -> imageView.setImageBitmap(mutableBitmap));
    }*/


    public void runObjectDetection(Bitmap bitmap) {
        if (imageCnt++>DROP_FRAME_COUNT) {
            imageCnt = 0;
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, true);

            TensorImage inputImageBuffer = TensorImage.fromBitmap(resizedBitmap);


            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.BILINEAR))
                    .add(new NormalizeOp(0f, 255f)) // Assuming your model expects inputs in [0, 1] or [-1, 1], adjust accordingly
                    .build();

            TensorImage processedImageBuffer = imageProcessor.process(inputImageBuffer);

            // Output tensor shape for YOLOv8 might vary. Common shape: [1, num_detections, 4 + num_classes]
            int[] outputShape = tflite.getOutputTensor(0).shape();
            TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32);

            /*         tflite.run(processedImageBuffer.getBuffer(), outputFeature0.getBuffer().rewind());
                       float[] rawDetections = outputFeature0.getFloatArray();
                        List<Recognition> recognitions = processRawDetections(rawDetections, bitmap.getWidth(), bitmap.getHeight());
                        List<Recognition> finalRecognitions = nonMaxSuppression(recognitions);
             */


            float[][][] output= new float[outputShape[0]][outputShape[1]][outputShape[2]];

            tflite.run(processedImageBuffer.getBuffer(), output);
            List<Recognition> recognitions = processRawDetections(output, bitmap.getWidth(), bitmap.getHeight());
            List<Recognition> finalRecognitions = nonMaxSuppression(recognitions);

            runOnUiThread(() -> imageView.setImageBitmap(drawDetections(bitmap, finalRecognitions)));
        }
    }

    private List<Recognition> processRawDetections(float[][][] rawDetections, int imageWidth, int imageHeight) {
        List<Recognition> recognitions = new ArrayList<>();
        float[][] rawRowCol=rawDetections[0];

        // Assuming the output shape is [1, num_detections, 4 + num_classes]
        int numDetections = tflite.getOutputTensor(0).shape()[2];
        int detectionStride = tflite.getOutputTensor(0).shape()[1];

        //numDetections = 10;

        for (int i = 0; i < numDetections; i++) {

            int offset = i * detectionStride;

            // The first 4 elements are likely [centerX, centerY, width, height] normalized to [0, 1]
            float centerX = rawRowCol[0][i];
            float centerY = rawRowCol[1][i];
            float width = rawRowCol[2][i];
            float height = rawRowCol[3][i];

            // The remaining elements are class probabilities
            for (int j = 0; j < numClasses; j++) {
                float confidence = rawRowCol[4+j][i];
                if (confidence > confidenceThreshold) {
                    String label = (j < labels.size()) ? labels.get(j) : "unknown";

                    // Scale the bounding box to the original image size
                    int x = (int) ((centerX - width / 2f) * imageWidth);
                    int y = (int) ((centerY - height / 2f) * imageHeight);
                    int w = (int) (width * imageWidth);
                    int h = (int) (height * imageHeight);

                    RectF location = new RectF(x, y, x + w, y + h);
                    recognitions.add(new Recognition("" + j, label, confidence, location));
                }
            }


/*
            // The remaining elements are class probabilities
            float maxConfidence = 0;
            int maxIndex = -1;
            for (int j = 0; j < numClasses; j++) {
                float confidence = rawRowCol[4 + j][i];
                if (confidence > confidenceThreshold) {
                    if (confidence > maxConfidence) {
                        maxConfidence = confidence;
                        maxIndex = j;
                    }
                }
            }
            if (maxIndex != -1) {
                String label = (maxIndex < labels.size()) ? labels.get(maxIndex) : "unknown";

                // Scale the bounding box to the original image size
                int x = (int) ((centerX - width / 2f) * imageWidth);
                int y = (int) ((centerY - height / 2f) * imageHeight);
                int w = (int) (width * imageWidth);
                int h = (int) (height * imageHeight);

                RectF location = new RectF(x, y, x + w, y + h);
                recognitions.add(new Recognition("" + maxIndex, label, maxConfidence, location));

            }


 */
        }
        return recognitions;
    }


    private List<Recognition> processRawDetections(float[] rawDetections, int imageWidth, int imageHeight) {
        List<Recognition> recognitions = new ArrayList<>();

        // Assuming the output shape is [1, num_detections, 4 + num_classes]
        int numDetections = tflite.getOutputTensor(0).shape()[2];
        int detectionStride = tflite.getOutputTensor(0).shape()[1];

        //numDetections = 10;

        for (int i = 0; i < numDetections; i++) {
            int offset = i * detectionStride;

            // The first 4 elements are likely [centerX, centerY, width, height] normalized to [0, 1]
            float centerX = rawDetections[offset + 0];
            float centerY = rawDetections[offset + 1];
            float width = rawDetections[offset + 2];
            float height = rawDetections[offset + 3];

            // The remaining elements are class probabilities
            float maxConfidence = 0;
            int maxIndex = -1;
            for (int j = 0; j < numClasses; j++) {
                float confidence = rawDetections[offset + 4 + j];
                if (confidence > confidenceThreshold) {
                    if (confidence > maxConfidence) {
                        maxConfidence = confidence;
                        maxIndex = j;
                    }
                }
            }
            if (maxIndex != -1) {
                String label = (maxIndex < labels.size()) ? labels.get(maxIndex) : "unknown";

                // Scale the bounding box to the original image size
                int x = (int) ((centerX - width / 2f) * imageWidth);
                int y = (int) ((centerY - height / 2f) * imageHeight);
                int w = (int) (width * imageWidth);
                int h = (int) (height * imageHeight);

                RectF location = new RectF(x, y, x + w, y + h);
                recognitions.add(new Recognition("" + maxIndex, label, maxConfidence, location));

            }
        }
        return recognitions;
    }

    private List<Recognition> nonMaxSuppression(List<Recognition> recognitions) {
        List<Recognition> filteredRecognitions = new ArrayList<>();
        if (recognitions.isEmpty()) return filteredRecognitions;

        // Sort detections by confidence in descending order
        recognitions.sort(Comparator.comparing(Recognition::getConfidence).reversed());

        boolean[] suppressed = new boolean[recognitions.size()];
        for (int i = 0; i < recognitions.size(); i++) {
            if (suppressed[i]) continue;
            filteredRecognitions.add(recognitions.get(i));
            for (int j = i + 1; j < recognitions.size(); j++) {
                if (suppressed[j] || recognitions.get(i).getLabel() != recognitions.get(j).getLabel()) continue;
                float iou = calculateIoU(recognitions.get(i).getLocation(), recognitions.get(j).getLocation());
                if (iou > nmsIoUThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        return filteredRecognitions;
    }

    private float calculateIoU(RectF box1, RectF box2) {
        float intersectionX = Math.max(box1.left, box2.left);
        float intersectionY = Math.max(box1.top, box2.top);
        float intersectionRight = Math.min(box1.right, box2.right);
        float intersectionBottom = Math.min(box1.bottom, box2.bottom);

        float intersectionArea = Math.max(0, intersectionRight - intersectionX) * Math.max(0, intersectionBottom - intersectionY);

        float area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
        float area2 = (box2.right - box2.left) * (box2.bottom - box2.top);

        return intersectionArea / (area1 + area2 - intersectionArea + 1e-5f); // Adding a small epsilon to avoid division by zero
    }

    private Bitmap drawDetections(Bitmap bitmap, List<Recognition> recognitions) {
        Bitmap outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(outputBitmap);
        Paint boxPaint = new Paint();
        boxPaint.setColor(Color.RED);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5);

        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextSize(50);

        for (Recognition recognition : recognitions) {
            RectF location = recognition.getLocation();
            String label = recognition.getLabel();
            float confidence = recognition.getConfidence();

            canvas.drawRect(location, boxPaint);
            String displayText = String.format("%s (%.2f)", label, confidence);
            canvas.drawText(displayText, location.left, location.top - 10, textPaint);
        }
        return outputBitmap;
    }

    public static class Recognition {
        private final String id;
        private final String label;
        private final Float confidence;
        private final RectF location;

        public Recognition(final String id, final String label, final Float confidence, final RectF location) {
            this.id = id;
            this.label = label;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getLabel() {
            return label;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return location;
        }

        @Override
        public String toString() {
            return "Recognition{" +
                    "id='" + id + '\'' +
                    ", label='" + label + '\'' +
                    ", confidence=" + confidence +
                    ", location=" + location +
                    '}';
        }
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}