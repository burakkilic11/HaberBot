# Python'un slim versiyonunu temel alalım (3.11 önerilir, kullandığın kütüphanelerle uyumlu olmalı)
FROM python:3.11-slim

# Ortam değişkenlerini ayarla
ENV RUNNING_IN_DOCKER=true
ENV PYTHONUNBUFFERED=1

# Çalışma dizinini ayarla
WORKDIR /app

# ---- YENİ ADIM: Build bağımlılıklarını yükle ----
# chroma-hnswlib gibi C++ eklentilerini derlemek için gerekli araçlar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # python3-dev paketi Python C API başlık dosyalarını sağlar (genellikle build-essential ile gelir ama emin olmak için eklenebilir)
    # python3-dev \
    && rm -rf /var/lib/apt/lists/*
# -----------------------------------------------

# Önce sadece requirements.txt dosyasını kopyala
COPY requirements.txt ./

# Bağımlılıkları yükle (pip'i güncelleyerek başla)
# --no-cache-dir kullanarak imaj boyutunu küçük tut
RUN pip install --no-cache-dir --upgrade pip
# Bu komut artık çalışmalı çünkü derleyici yüklendi
RUN pip install --no-cache-dir -r requirements.txt

# Şimdi uygulamanın geri kalanını (kod ve veritabanı) kopyala
COPY app.py ./
# ChromaDB veritabanını olduğu gibi kopyala
COPY chroma_db ./chroma_db/

# Streamlit tarafından kullanılacak varsayılan portu belirt
EXPOSE 8501

# Container başladığında çalıştırılacak komut
# Streamlit'i dışarıdan erişilebilir şekilde başlat (0.0.0.0)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]