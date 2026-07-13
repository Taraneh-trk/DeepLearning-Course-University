# گزارش پیاده‌سازی پروژه سودوکو با PyTorch

## نوع مدل استفاده شده

**مدل از نوع CNN (Convolutional Neural Network) است.**

معماری مدل (مشابه نسخه Keras):
```
Conv2D(1, 32, kernel_size=3, activation='relu')
Conv2D(32, 64, kernel_size=3, activation='relu')
MaxPool2D(pool_size=(2,2))
Dropout(0.25)
Flatten()
Dense(128, activation='relu')
Dropout(0.5)
Dense(9, activation='softmax')
```

---

## تبدیل مدل از TensorFlow/Keras به PyTorch

برای تبدیل مدل آموزش‌دیده از Keras به PyTorch، از روش **انتقال مستقیم وزن‌ها** استفاده شده است. کد تبدیل به گونه‌ای نوشته شده که در صورت نبودن فایل تبدیل‌شده (`.pth`)، به طور خودکار فایل `.h5` را خوانده و وزن‌ها را تبدیل کرده و فایل جدید را می‌سازد.

### مراحل انجام شده:

1. **بازسازی معماری مدل در هر دو فریمورک**  
   ساختار دقیقاً یکسان ایجاد شد (تعداد لایه‌ها، فیلترها، کرنل‌ها، activation functions).  
   در PyTorch، مدل با استفاده از دو `nn.Sequential` به نام‌های `features` و `classifier` پیاده‌سازی شده است.

2. **بارگذاری وزن‌های Keras**  
   ```python
   keras_model.load_weights('digitRecognition.h5')
   ```

3. **تبدیل وزن‌های لایه‌های Conv2D**  
   تفاوت در ترتیب ذخیره‌سازی ابعاد:
   - Keras: `[kernel_height, kernel_width, in_channels, out_channels]`
   - PyTorch: `[out_channels, in_channels, kernel_height, kernel_width]`

   بنابراین از `transpose(3, 2, 0, 1)` استفاده می‌شود.

   کد تبدیل برای لایه اول (ایندکس 0 در Keras):
   ```python
   state_dict['features.0.weight'] = torch.tensor(np.transpose(w, (3, 2, 0, 1)).copy())
   state_dict['features.0.bias']   = torch.tensor(b.copy())
   ```

   برای لایه دوم (ایندکس 1 در Keras) با توجه به ساختار `nn.Sequential`، ایندکس لایه `features.2` است (چون بین آن‌ها یک `ReLU` وجود دارد):
   ```python
   state_dict['features.2.weight'] = torch.tensor(np.transpose(w, (3, 2, 0, 1)).copy())
   state_dict['features.2.bias']   = torch.tensor(b.copy())
   ```

4. **تبدیل وزن‌های لایه‌های Dense**  
   نیاز به transpose ساده (چون Keras ذخیره می‌کند `[input_dim, output_dim]` در حالی که PyTorch انتظار `[output_dim, input_dim]` دارد):
   ```python
   # لایه Dense اول (ایندکس 5 در Keras) → classifier.0
   state_dict['classifier.0.weight'] = torch.tensor(np.transpose(w).copy())
   state_dict['classifier.0.bias']   = torch.tensor(b.copy())

   # لایه Dense خروجی (ایندکس 7 در Keras) → classifier.3
   state_dict['classifier.3.weight'] = torch.tensor(np.transpose(w).copy())
   state_dict['classifier.3.bias']   = torch.tensor(b.copy())
   ```

5. **ذخیره مدل PyTorch**  
   ```python
   torch.save(pytorch_model.state_dict(), 'digitRecognition_pytorch_converted.pth')
   ```

6. **بارگذاری در برنامه اصلی**  
   در فایل `main_pytorch.py` و `RealTimeSudokuSolver_pytorch.py`، مدل از فایل مشترک `sudoku_model.py` وارد شده و وزن‌ها بارگذاری می‌شوند.

---

## مشکلات پیش آمده و راه حل‌ها

### مشکل 1: عدم تطابق ترتیب ابعاد در لایه‌های Conv2D

**مشکل:** هنگام کپی مستقیم وزن‌ها، خطای شکل (shape mismatch) دریافت می‌شد.

**راه حل:** استفاده از `transpose(3, 2, 0, 1)` برای تبدیل ترتیب ابعاد از فرمت Keras به PyTorch.

---

### مشکل 2: تفاوت در ترتیب کانال ورودی

**مشکل:** Keras از `(height, width, channels)` و PyTorch از `(channels, height, width)` استفاده می‌کند.

**راه حل:** در تابع پیش‌پردازش (`RealTimeSudokuSolver_pytorch.py`)، تصویر استخراج شده به شکل `(1, 1, 28, 28)` تبدیل می‌شود (batch, channel, height, width) و سپس به مدل داده می‌شود.

---

### مشکل 3: اشتباه در ایندکس لایه‌های `nn.Sequential` هنگام تبدیل

**مشکل:** در ابتدا تصور می‌شد لایه دوم Conv2D با ایندکس `features.3` ذخیره شود، در حالی که با توجه به ساختار:
```
features.0 : Conv2d
features.1 : ReLU
features.2 : Conv2d   ← لایه دوم اینجاست
features.3 : ReLU
...
```
در نتیجه هنگام بارگذاری، خطای `Unexpected key(s): "features.3.weight"` رخ می‌داد.

**راه حل:** اصلاح ایندکس لایه دوم به `features.2.weight` و `features.2.bias`. همچنین برای لایه‌های `classifier` نیز ایندکس‌ها به‌درستی تعیین شدند:
- `classifier.0` : Linear اول
- `classifier.1` : ReLU
- `classifier.2` : Dropout
- `classifier.3` : Linear آخر

---

### مشکل 4: عدم وجود فایل تبدیل‌شده هنگام اجرای اولیه

**مشکل:** برنامه اصلی برای بارگذاری مدل به فایل `digitRecognition_pytorch_converted.pth` نیاز داشت، اما این فایل ابتدا وجود نداشت.

**راه حل:** در کد تبدیل، شرطی تعبیه شد که اگر فایل `.pth` وجود نداشته باشد، به طور خودکار با استفاده از TensorFlow وزن‌ها را از `digitRecognition.h5` خوانده، تبدیل کرده و فایل را می‌سازد. سپس مدل از همان فایل بارگذاری می‌شود. این کار باعث می‌شود پروژه بدون نیاز به مرحله دستی تبدیل، آماده اجرا باشد.

---

### مشکل 5: وابستگی متقابل تعریف مدل در فایل‌های مختلف

**مشکل:** کلاس مدل در سه فایل `digitRecognition_pytorch.py`، `main_pytorch.py` و `RealTimeSudokuSolver_pytorch.py` تکرار شده بود که باعث می‌شد هر تغییری باید در چند جا اعمال شود.

**راه حل:** ایجاد فایل مشترک `sudoku_model.py` که تنها یک بار کلاس `SudokuCNN` را تعریف می‌کند. سایر فایل‌ها با دستور `from sudoku_model import SudokuCNN` از آن استفاده می‌کنند. همچنین در فایل آموزش، کل کد درون `if __name__ == "__main__":` قرار گرفته تا هنگام import شدن اجرا نشود.

---

## نتیجه نهایی

پس از اعمال اصلاحات، مدل PyTorch با موفقیت وزن‌های مدل Keras را بارگذاری می‌کند و در برنامه زمان واقعی (وب‌کم) بدون خطا عمل می‌نماید. دقت تشخیص اعداد با نسخه اصلی Keras برابر است (چون وزن‌ها دقیقاً منتقل شده‌اند). همچنین فرآیند تبدیل خودکار، تجربه کاربری را ساده‌تر کرده است.