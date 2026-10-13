package com.scansklad.ocrlab;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;

/** Recognizer-only PaddleOCR PP-OCRv5 Cyrillic ONNX. Feed exactly ONE cropped text line. */
public final class PaddleLineRecognizer implements AutoCloseable {
    private static final int HEIGHT = 48;
    private static final int WIDTH = 320;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final OrtSession.SessionOptions options;
    private final String inputName;
    private final List<String> alphabet;

    public static final class Recognition {
        public final String text;
        public final float confidence;
        public final long millis;
        Recognition(String text, float confidence, long millis) {
            this.text = text; this.confidence = confidence; this.millis = millis;
        }
    }

    public PaddleLineRecognizer(Context ctx) throws Exception {
        alphabet = loadAlphabet(ctx);
        env = OrtEnvironment.getEnvironment();
        byte[] model;
        try (InputStream in = ctx.getAssets().open("rec/inference.onnx")) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] block = new byte[65536]; int n;
            while ((n = in.read(block)) != -1) bytes.write(block, 0, n);
            model = bytes.toByteArray();
        }
        options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(2);
        options.setInterOpNumThreads(1);
        session = env.createSession(model, options);
        inputName = session.getInputNames().iterator().next();
    }

    /** Gets character_dict from *the model's own* inference.yml (not a generic OCR alphabet). */
    private static List<String> loadAlphabet(Context ctx) throws Exception {
        List<String> chars = new ArrayList<>();
        boolean inList = false;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                ctx.getAssets().open("rec/inference.yml"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                String s = line.trim();
                if (s.equals("character_dict:")) { inList = true; continue; }
                if (!inList) continue;
                if (!s.startsWith("- ")) break;
                String label = s.substring(2).trim();
                if (label.startsWith("'") && label.endsWith("'") && label.length() >= 2) {
                    label = label.substring(1, label.length() - 1).replace("''", "'");
                } else if (label.startsWith("\"") && label.endsWith("\"") && label.length() >= 2) {
                    label = label.substring(1, label.length() - 1).replace("\\\"", "\"");
                }
                chars.add(label);
            }
        }
        if (chars.size() < 100 || !chars.contains("Л") || !chars.contains("П"))
            throw new IllegalStateException("Incorrect Cyrillic dictionary: " + chars.size() + " symbols");
        return chars;
    }

    public synchronized Recognition recognize(Bitmap targetRoi) throws Exception {
        long startNs = System.nanoTime();
        Bitmap line = trimDarkRows(targetRoi);
        float[] chw = toTensor(line);
        if (line != targetRoi) line.recycle();
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1, 3, HEIGHT, WIDTH});
             OrtSession.Result outputs = session.run(Collections.singletonMap(inputName, in))) {
            Object value = ((OnnxTensor)outputs.get(0)).getValue();
            if (!(value instanceof float[][][])) {
                throw new IllegalStateException("Unexpected ONNX output type: " + value.getClass().getName());
            }
            float[][][] probs = (float[][][]) value;
            if (probs.length != 1 || probs[0].length == 0) throw new IllegalStateException("Invalid output shape");
            int classes = probs[0][0].length;
            // PP-OCRv5 Cyrillic's exported CTC head has a trailing SPACE class.
            // The model's inference.yml lists 850 characters, but the head emits
            // 852 classes: CTC blank (0), the 850 YAML chars (1..850), SPACE (851).
            // Add only the final SPACE; never shift the existing character IDs.
            // Keep rejecting any other shape/dictionary mismatch instead of
            // silently producing potentially incorrect warehouse addresses.
            if (classes == alphabet.size() + 2 && !alphabet.contains(" ")) {
                alphabet.add(" ");
            }
            if (classes != alphabet.size() + 1) {
                throw new IllegalStateException(String.format(Locale.ROOT,
                        "Character dictionary mismatch: model=%d, yaml=%d+blank", classes, alphabet.size()));
            }
            StringBuilder output = new StringBuilder();
            int last = -1, accepted = 0;
            double score = 0;
            for (float[] timestep : probs[0]) {
                int bestIndex = 0; float bestProb = -Float.MAX_VALUE;
                for (int i = 0; i < timestep.length; i++) {
                    if (timestep[i] > bestProb) { bestProb = timestep[i]; bestIndex = i; }
                }
                if (bestIndex > 0 && bestIndex != last) {
                    output.append(alphabet.get(bestIndex - 1));
                    accepted++;
                    score += bestProb;
                }
                last = bestIndex;
            }
            long elapsed = (System.nanoTime() - startNs) / 1_000_000L;
            return new Recognition(output.toString(), accepted == 0 ? 0f : (float) (score / accepted), elapsed);
        }
    }

    private static float[] toTensor(Bitmap original) {
        float[] out = new float[3 * HEIGHT * WIDTH]; // Right padding is normalized 0, as in RecResizeImg.
        int resizedWidth = Math.min(WIDTH, Math.max(1,
                (int)Math.ceil((double) original.getWidth() * HEIGHT / original.getHeight())));
        Bitmap bitmap = Bitmap.createScaledBitmap(original, resizedWidth, HEIGHT, true);
        int[] pixels = new int[resizedWidth * HEIGHT];
        bitmap.getPixels(pixels, 0, resizedWidth, 0, 0, resizedWidth, HEIGHT);
        if (bitmap != original) bitmap.recycle();
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < resizedWidth; x++) {
                int rgb = pixels[y * resizedWidth + x];
                int i = y * WIDTH + x;
                out[i] = (Color.blue(rgb) / 255f - .5f) / .5f; // BGR as PaddleOCR's DecodeImage
                out[HEIGHT * WIDTH + i] = (Color.green(rgb) / 255f - .5f) / .5f;
                out[2 * HEIGHT * WIDTH + i] = (Color.red(rgb) / 255f - .5f) / .5f;
            }
        }
        return out;
    }

    /** Remove broad top/bottom white margins; do not crop x or replace ambiguous letters. */
    private static Bitmap trimDarkRows(Bitmap src) {
        int w = src.getWidth(), h = src.getHeight();
        if (w < 30 || h < 20) return src;
        int step = Math.max(1, w / 400);
        int min = h, max = 0;
        for (int y = Math.max(0, h / 20); y < h - h / 20; y++) {
            int count = 0, total = 0;
            for (int x = w / 30; x < w - w / 30; x += step) {
                int c = src.getPixel(x, y);
                int gray = (Color.red(c) * 299 + Color.green(c) * 587 + Color.blue(c) * 114) / 1000;
                if (gray < 125) count++;
                total++;
            }
            if (count > total * .008) { min = Math.min(min, y); max = Math.max(max, y); }
        }
        int active = max - min;
        // Skip trimming on noisy/dark backgrounds and nearly-empty frames.
        if (min >= max || active < h * .035 || active > h * .83) return src;
        int pad = Math.max(4, active / 5);
        min = Math.max(0, min - pad);
        max = Math.min(h, max + pad);
        return Bitmap.createBitmap(src, 0, min, w, max - min);
    }

    @Override public void close() throws OrtException {
        try { session.close(); } finally { options.close(); }
    }
}
