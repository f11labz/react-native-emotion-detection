package com.reactnativeemotiondetection.face;

import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Callback;
import com.reactnativeemotiondetection.face.predictivemodels.Classification;
import com.reactnativeemotiondetection.face.predictivemodels.TensorFlowClassifier;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.util.Base64;
import android.util.Log;
import android.widget.Toast;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

public class FaceModule extends ReactContextBaseJavaModule {
    private static ReactApplicationContext reactContext;
    TensorFlowClassifier classifier;
    static final int PIXEL_WIDTH = 48;

    // age code
    private static final String MODEL_FILE = "file:///android_asset/frozen_age_graph.pb";
    private static final String MODEL_LABELS = "file:///android_asset/age_labels.txt";
    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "output/output";
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final int MAX_FACES=6;
    private static final int texBackgroundWaiting = Color.parseColor("#ccff0000");
    private static final int texBackground = Color.parseColor("#ccffffff");
    //MAX_BATCH_SZ = 128
    //AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
    private Vector<String> labels = new Vector<String>();
    //RESIZE_FINAL = 227
    private static final int IMAGE_INPUT_SIZE = 227;
    //Color Image? or grayscale and normalized?
    private static final int[] INPUT_SIZE = {IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3};
    private TensorFlowInferenceInterface inferenceInterface;

    FaceModule(ReactApplicationContext context) {
        super(context);
        this.reactContext = context;
    }

    @Override
    public String getName() {
        return "FaceModule";
    }

    @Override
    public Map<String, Object> getConstants() {
        final Map<String, Object> constants = new HashMap<>();
        return constants;
    }

    @ReactMethod
    public void loadModel() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier=TensorFlowClassifier.create(reactContext.getAssets(), "CNN",
                            "opt_em_convnet_5000.pb", "labels.txt", PIXEL_WIDTH,
                            "input", "output_50", true, 7);
                    String actualFilename = MODEL_LABELS.split("file:///android_asset/")[1];
                    Log.i("FACE_MODULE", "Reading labels from: " + actualFilename);
                    BufferedReader br = null;
                    try {
                        br = new BufferedReader(new InputStreamReader(reactContext.getAssets().open(actualFilename)));
                        String line;
                        while ((line = br.readLine()) != null) {
                            labels.add(line);
                        }
                        br.close();
                    } catch (IOException e) {
                        throw new RuntimeException("Problem reading label file!" , e);
                    }

                    Log.i("OnCreate", "Labels:" + labels.toArray().toString());
                    String modelActualFilename = MODEL_FILE.split("file:///android_asset/")[1];
                    Log.i("onCreate", "Reading model from: " + modelActualFilename);

                    inferenceInterface = new TensorFlowInferenceInterface(reactContext.getAssets(), MODEL_FILE);
                    // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
                    final Operation operation = inferenceInterface.graphOperation(OUTPUT_NODE);
                    final int numClasses = (int) operation.output(0).shape().size(1);
                    Log.i("FACE_MODULE", "Read " + labels.size() + " labels, output layer size is " + numClasses);

                } catch (final Exception e) {
                    //if they aren't found, throw an error!
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }

    @ReactMethod
    public void detectEmotion(String path, final Callback callback){
        byte[] decodedString = Base64.decode(path, Base64.DEFAULT);
        Bitmap image = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
        Bitmap grayImage = toGrayscale(image);
        Bitmap resizedImage=getResizedBitmap(grayImage,48,48);
        int pixelarray[];

        //Initialize the intArray with the same size as the number of pixels on the image
        pixelarray = new int[resizedImage.getWidth()*resizedImage.getHeight()];
        //copy pixel data from the Bitmap into the 'intArray' array
        resizedImage.getPixels(pixelarray, 0, resizedImage.getWidth(), 0, 0, resizedImage.getWidth(), resizedImage.getHeight());

        float normalized_pixels [] = new float[pixelarray.length];
        for (int i=0; i < pixelarray.length; i++) {
            // 0 for white and 255 for black
            int pix = pixelarray[i];
            int b = pix & 0xff;
            //  normalized_pixels[i] = (float)((0xff - b)/255.0);
            // normalized_pixels[i] = (float)(b/255.0);
            normalized_pixels[i] = (float)(b);

        }
        String text=null;
        try{
            final Classification res = classifier.recognize(normalized_pixels);
            //if it can't classify, output a question mark
            if (res.getLabel() == null) {
                text = "Status: "+ ": ?\n";
            } else {
                //else output its name
                text = String.format("%s: %s, %f\n", "Status: ", res.getLabel(),
                        res.getConf());
            }
            callback.invoke(null, res.getLabel());
        }
        catch (Exception  e) {
            System.out.print("Exception:" + e.toString());
            callback.invoke(e.toString(), null);
        }
    }

    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }
    public Bitmap getResizedBitmap(Bitmap image, int bitmapWidth, int bitmapHeight) {
        return Bitmap.createScaledBitmap(image, bitmapWidth, bitmapHeight, true);
    }
    private static Bitmap resize(Bitmap image, int maxWidth, int maxHeight) {
        if (maxHeight > 0 && maxWidth > 0) {
            int width = image.getWidth();
            int height = image.getHeight();
            float ratioBitmap = (float) width / (float) height;
            float ratioMax = (float) maxWidth / (float) maxHeight;

            int finalWidth = maxWidth;
            int finalHeight = maxHeight;
            if (ratioMax > ratioBitmap) {
                finalWidth = (int) ((float)maxHeight * ratioBitmap);
            } else {
                finalHeight = (int) ((float)maxWidth / ratioBitmap);
            }
            image = Bitmap.createScaledBitmap(image, finalWidth, finalHeight, true);
            return image;
        } else {
            return image;
        }
    }
}
