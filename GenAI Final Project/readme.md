Proje ana klasörüne .env oluştur. NEWSDATA_API_KEY="" Şeklinde yazarak newsdata api keyi belirt.

Projeyi lokalde çalıştırmak için:

Terminale: streamlit run app.py 
yaz ve çalıştır

Projeyi Docker'da çalıştırmak için build aldıktan sonra:

Ollamayı 0.0.0.0'a açmak için: 

Powershell yönetici olarak çalıştır, 
$env:OLLAMA_HOST="0.0.0.0"                                                                      
>> ollama serve   
Kodu çalıştır.

Projeyi aç Terminale yaz ve çalıştır:
docker run --rm -p 8501:8501 -e NEWSDATA_API_KEY="TYPE YOUR API KEY HERE" -e OLLAMA_HOST="http://host.docker.internal:11434" --name my-rag-container rag-chatbot-app

Bu URL'i kullanarak chatbotu kullan:
http://localhost:8501
