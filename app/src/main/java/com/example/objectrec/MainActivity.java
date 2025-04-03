package com.example.objectrec;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.ViewGroup;
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
import java.util.Arrays;
import java.util.List;
import androidx.camera.view.PreviewView;

public class MainActivity extends AppCompatActivity {

    private PreviewView previewView;

    private static final String TAG = "MainActivity";
    private static final int REQUEST_CAMERA_PERMISSION = 1;
    private TextureView textureView;
    private ImageView imageView;
    private Interpreter tflite;
    private int imageSizeX;
    private int imageSizeY;
    private final List<String> labels= Arrays.asList("A","B","C","D");
    private Handler handler;
    private HandlerThread handlerThread;

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
/*        try {
            tflite = new Interpreter(FileUtil.loadMappedFile(this, "your_model.tflite")); // Replace with your model file name
            labels = FileUtil.loadLabels(this, "your_labels.txt"); // Replace with your labels file name

            int[] inputShape = tflite.getInputTensor(0).shape();
            imageSizeY = inputShape[1];
            imageSizeX = inputShape[2];



        } catch (IOException e) {
            e.printStackTrace();
        }

        */
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(imageSizeX, imageSizeY))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), imageProxy -> {
                    Bitmap bitmap = toBitmap(imageProxy);
                    if (bitmap != null) {
                        handler.post(() -> runObjectDetection(bitmap));
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

    private Bitmap toBitmap(ImageProxy image) {
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        android.graphics.YuvImage yuvImage = new android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, image.getWidth(), image.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private void runObjectDetection(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, true);
        TensorImage inputImageBuffer = TensorImage.fromBitmap(resizedBitmap);

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(imageSizeY, imageSizeX, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0, 255))
                .build();
/*
        TensorImage processedImageBuffer = imageProcessor.process(inputImageBuffer);

        TensorBuffer outputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 10, 4}, DataType.FLOAT32);
        TensorBuffer outputFeature1 = TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.FLOAT32);
        TensorBuffer outputFeature2 = TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.FLOAT32);


        tflite.run(processedImageBuffer.getBuffer(), new Object[]{outputFeature0.getBuffer().rewind(), outputFeature1.getBuffer().rewind(), outputFeature2.getBuffer().rewind()});

        float[] locations = outputFeature0.getFloatArray(); // Corrected line.
        float[] classes = outputFeature1.getFloatArray();
        float[] scores = outputFeature2.getFloatArray();
        */

        float[] locations = new float[]{0.1f, 0.2f, 0.8f, 0.9f, 0.5f, 0.6f, 0.7f, 0.8f
                , 0.9f, 1.0f, 1.1f, 1.2f, 0, 0, 0, 0};
        float[] classes = new float[]{0, 1, 2, 3};
        float[] scores = new float[]{0.6f, 0.2f, 0.3f, 0.1f};

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