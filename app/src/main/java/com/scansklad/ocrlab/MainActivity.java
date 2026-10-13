package com.scansklad.ocrlab;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/** Separate offline comparison app. It does NOT read/write ScanSklad v2.0's data. */
public final class MainActivity extends AppCompatActivity {
    private PreviewView cameraView;
    private View scanFrame;
    private TextView status, paddleResult, mlResult, testCount;
    private EditText expected;
    private Button scanButton, autoButton;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ExecutorService inferenceWorker = Executors.newSingleThreadExecutor();
    private PaddleLineRecognizer paddle;
    private TextRecognizer mlkit;
    private volatile boolean processing = false;
    private boolean autoMode = false;
    private boolean active = false;
    private int modelFailures = 0;
    private ProcessCameraProvider cameraProvider;
    private String lastPaddle = "", lastMl = "";
    private long lastPaddleMs = 0, lastMlMs = 0;
    private StringBuilder report = new StringBuilder("ScanSklad OCR LAB — сравнительный тест\n");
    private int reports = 0;
    private final Runnable autoScan = new Runnable() {
        @Override public void run() {
            if (!active) return;
            if (autoMode) captureAndCompare();
            ui.postDelayed(this, 1400);
        }
    };
    private final ActivityResultLauncher<String> permission =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), ok -> {
                if (Boolean.TRUE.equals(ok)) startPreview();
                else status.setText("Для тестирования нужна камера. Разрешите её в настройках Android.");
            });

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_main);
        cameraView = findViewById(R.id.previewView);
        cameraView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        scanFrame = findViewById(R.id.targetFrame);
        status = findViewById(R.id.statusText);
        paddleResult = findViewById(R.id.paddleResult);
        mlResult = findViewById(R.id.mlResult);
        testCount = findViewById(R.id.countText);
        expected = findViewById(R.id.expectedInput);
        scanButton = findViewById(R.id.scanButton);
        autoButton = findViewById(R.id.autoButton);
        mlkit = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        reports = getPreferences(MODE_PRIVATE).getInt("report_count", 0);
        report = new StringBuilder(getPreferences(MODE_PRIVATE).getString("report", "ScanSklad OCR LAB — сравнительный тест\n"));
        testCount.setText("Записано тестов: " + reports);
        scanButton.setOnClickListener(v -> captureAndCompare());
        autoButton.setOnClickListener(v -> {
            autoMode = !autoMode;
            autoButton.setText(autoMode ? "АВТО: ВКЛ" : "АВТО: ВЫКЛ");
        });
        findViewById(R.id.saveButton).setOnClickListener(v -> recordSample());
        findViewById(R.id.shareButton).setOnClickListener(v -> shareReport());

        inferenceWorker.submit(() -> {
            try {
                PaddleLineRecognizer built = new PaddleLineRecognizer(getApplicationContext());
                paddle = built;
                runOnUiThread(() -> status.setText("Модель PaddleOCR загружена. Наведите рамку на строку."));
            } catch (Throwable failure) {
                runOnUiThread(() -> status.setText("Ошибка PaddleOCR: " + failure.getClass().getSimpleName()
                        + " — " + failure.getMessage()));
            }
        });
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startPreview();
        else permission.launch(Manifest.permission.CAMERA);
    }

    private void startPreview() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(cameraView.getSurfaceProvider());
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview);
            } catch (Exception e) {
                status.setText("Не удалось запустить камеру: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void captureAndCompare() {
        if (processing || paddle == null || !active) return;
        Bitmap snapshot = cameraView.getBitmap();
        if (snapshot == null) { status.setText("Камера ещё не готова. Попробуйте ещё раз."); return; }
        // PreviewView.getBitmap() is the already transformed/scaled image which the user sees.
        // The red frame is a sibling of the PreviewView in the same FrameLayout.
        float mx = (float) snapshot.getWidth() / Math.max(1, cameraView.getWidth());
        float my = (float) snapshot.getHeight() / Math.max(1, cameraView.getHeight());
        int l = Math.max(0, Math.round((scanFrame.getLeft() - cameraView.getLeft()) * mx));
        int t = Math.max(0, Math.round((scanFrame.getTop() - cameraView.getTop()) * my));
        int r = Math.min(snapshot.getWidth(), Math.round((scanFrame.getRight() - cameraView.getLeft()) * mx));
        int b = Math.min(snapshot.getHeight(), Math.round((scanFrame.getBottom() - cameraView.getTop()) * my));
        if (r - l < 50 || b - t < 25) { snapshot.recycle(); return; }
        Bitmap roi = Bitmap.createBitmap(snapshot, l, t, r-l, b-t);
        snapshot.recycle();
        processing = true;
        lastPaddle = ""; lastMl = ""; lastPaddleMs = 0; lastMlMs = 0;
        paddleResult.setText("Распознаём...");
        mlResult.setText("Распознаём...");
        status.setText("Распознаём один и тот же кадр двумя движками...");
        final AtomicInteger pending = new AtomicInteger(2);
        Runnable finish = () -> {
            if (pending.decrementAndGet() == 0) {
                roi.recycle();
                processing = false;
            }
        };

        inferenceWorker.execute(() -> {
            try {
                PaddleLineRecognizer.Recognition rec = paddle.recognize(roi);
                runOnUiThread(() -> {
                    lastPaddle = rec.text;
                    lastPaddleMs = rec.millis;
                    paddleResult.setText((rec.text.isEmpty() ? "[не распознано]" : rec.text)
                            + String.format(Locale.ROOT, "\n%.0f%% · %d мс", rec.confidence*100, rec.millis));
                    status.setText("Готово. Сверьте оба результата с этикеткой.");
                });
            } catch (Throwable e) {
                modelFailures++;
                runOnUiThread(() -> {
                    lastPaddle = "ОШИБКА: " + e.getMessage();
                    paddleResult.setText(lastPaddle);
                    status.setText("Ошибка модели — сохраните отчёт. Попытка " + modelFailures);
                });
            } finally { runOnUiThread(finish); }
        });

        long mlStart = SystemClock.elapsedRealtime();
        mlkit.process(InputImage.fromBitmap(roi, 0))
                .addOnSuccessListener(result -> {
                    lastMl = result.getText().replace('\n', ' ').trim();
                    lastMlMs = SystemClock.elapsedRealtime() - mlStart;
                    mlResult.setText((lastMl.isEmpty() ? "[не распознано]" : lastMl) + "\n" + lastMlMs + " мс");
                })
                .addOnFailureListener(err -> mlResult.setText("Ошибка ML Kit: " + err.getMessage()))
                .addOnCompleteListener(task -> finish.run());
    }

    /** Enter the ground truth from the label; store OCR outputs without substitutions. */
    private void recordSample() {
        String truth = expected.getText().toString().trim();
        if (truth.isEmpty()) {
            Toast.makeText(this, "Введите правильный адрес с этикетки", Toast.LENGTH_SHORT).show();
            return;
        }
        if (processing) {
            Toast.makeText(this, "Дождитесь окончания распознавания", Toast.LENGTH_SHORT).show();
            return;
        }
        reports++;
        String target = canonical(truth);
        boolean p = canonical(lastPaddle).equals(target);
        boolean m = canonical(lastMl).equals(target);
        report.append("\n#").append(reports).append(" Этикетка: ").append(truth)
                .append("\n PaddleOCR: ").append(lastPaddle).append(" [")
                .append(p ? "верно" : "ошибка").append("] ").append(lastPaddleMs).append(" мс")
                .append("\n ML Kit: ").append(lastMl).append(" [")
                .append(m ? "верно" : "ошибка").append("] ").append(lastMlMs).append(" мс\n");
        getPreferences(MODE_PRIVATE).edit().putInt("report_count", reports)
                .putString("report", report.toString()).apply();
        testCount.setText("Записано тестов: " + reports + "  |  последний: Paddle "
                + (p ? "✓" : "✗") + ", ML Kit " + (m ? "✓" : "✗"));
        Toast.makeText(this, "Тест записан", Toast.LENGTH_SHORT).show();
    }

    private static String canonical(String value) {
        return value.toUpperCase(Locale.ROOT).replaceAll("\\s+", "")
                .replace('∕','/').replace('／','/');
    }

    private void shareReport() {
        if (reports == 0) { Toast.makeText(this, "Сначала сохраните хотя бы один тест", Toast.LENGTH_SHORT).show(); return; }
        Intent send = new Intent(Intent.ACTION_SEND);
        send.setType("text/plain");
        send.putExtra(Intent.EXTRA_TEXT, report.toString());
        startActivity(Intent.createChooser(send, "Отправить результаты сравнения OCR"));
    }

    @Override protected void onResume() {
        super.onResume();
        active = true;
        ui.removeCallbacks(autoScan);
        ui.postDelayed(autoScan, 1400);
    }
    @Override protected void onPause() {
        active = false;
        ui.removeCallbacks(autoScan);
        super.onPause();
    }
    @Override protected void onDestroy() {
        if (cameraProvider != null) cameraProvider.unbindAll();
        inferenceWorker.execute(() -> {
            try { if (paddle != null) paddle.close(); } catch (Exception ignored) { }
        });
        inferenceWorker.shutdown();
        mlkit.close();
        super.onDestroy();
    }
}
