import streamlit as st
import os
import chromadb
import ollama
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
import json # For potential structured output parsing
from newsdataapi import NewsDataApiClient

load_dotenv()
# --- Ayarlar ---
NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY") 
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "resmi_gazete_bge_m3"
OLLAMA_EMBED_MODEL = "bge-m3:latest"
OLLAMA_LLM = "deepseek-r1:14b" # 'ollama list' ile kontrol ettim, model ismi bu şekilde olmalı

# --- Ollama Erişimi İçin Host Ayarı (Docker ve Lokal Çalıştırma İçin) ---
# Docker içinden host makinedeki Ollama'ya erişim için kullanılır.
# Docker run komutunda -e OLLAMA_HOST="http://<YOUR_HOST_IP>:11434" veya
# Mac/Win için -e OLLAMA_HOST="http://host.docker.internal:11434" şeklinde ayarlanabilir.
# Ortam değişkeni yoksa localhost varsayılır (lokal çalıştırma için).
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# Docker'da çalışıp çalışmadığını anlamak için bir ortam değişkeni kontrolü
# Dockerfile'a 'ENV RUNNING_IN_DOCKER=true' ekleyebiliriz.
IS_RUNNING_IN_DOCKER = os.environ.get("RUNNING_IN_DOCKER", "false").lower() == "true"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST")

if IS_RUNNING_IN_DOCKER and not OLLAMA_BASE_URL:
    # Docker'da çalışıyor ama host belirtilmemişse Mac/Win için varsayılanı dene
    OLLAMA_BASE_URL = "http://host.docker.internal:11434"
    print("Docker içinde çalışılıyor, Ollama host belirtilmedi. 'http://host.docker.internal:11434' deneniyor.")
elif not OLLAMA_BASE_URL:
    # Docker'da değil ve host belirtilmemişse localhost kullan
    OLLAMA_BASE_URL = DEFAULT_OLLAMA_HOST
    print(f"Ollama host belirtilmedi, varsayılan kullanılıyor: {OLLAMA_BASE_URL}")

print(f"Kullanılacak Ollama Adresi: {OLLAMA_BASE_URL}")


# --- ChromaDB ve Modelleri Başlatma ---

# Embedding Modeli
try:
    # base_url parametresini ekledik
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    print("Ollama Embeddings modeli başarıyla yüklendi.")
except Exception as e:
    st.error(f"Ollama Embedding modeli ({OLLAMA_EMBED_MODEL}) yüklenirken hata: {e}\n"
             f"Ollama Adresi: {OLLAMA_BASE_URL}\n"
             "Lütfen Ollama'nın çalıştığından, modelin indirildiğinden ve belirtilen adresten erişilebilir olduğundan emin olun.")
    st.stop()

# LLM Modeli
try:
    # base_url parametresini ekledik
    llm = ChatOllama(model=OLLAMA_LLM, temperature=0, base_url=OLLAMA_BASE_URL)
    # Test sorgusu (isteğe bağlı, başlatma sırasında hata almamak için yorumda bırakılabilir)
    # try:
    #     llm.invoke("Test")
    # except Exception as test_e:
    #     print(f"LLM test sorgusu başarısız: {test_e}")
    #     raise RuntimeError(f"LLM test sorgusu başarısız: {test_e}")
    print(f"Ollama LLM ({OLLAMA_LLM}) başarıyla yüklendi.")
except Exception as e:
    st.error(f"Ollama LLM ({OLLAMA_LLM}) yüklenirken/test edilirken hata: {e}\n"
             f"Ollama Adresi: {OLLAMA_BASE_URL}\n"
             "Lütfen Ollama'nın çalıştığından, modelin indirildiğinden ve belirtilen adresten erişilebilir olduğundan emin olun.")
    st.stop()


# ChromaDB İstemcisi ve Vektör Deposu
try:
    if not os.path.exists(CHROMA_DB_PATH):
         raise FileNotFoundError(f"ChromaDB yolu bulunamadı: {CHROMA_DB_PATH}. "
                                 "Lütfen 'chroma_db' klasörünün bu betikle aynı dizinde olduğundan emin olun.")
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings # embedding_function'ı burada belirtmek önemli
    )
    # Koleksiyonun boş olup olmadığını kontrol et
    if vectorstore._collection.count() == 0:
         st.warning(f"UYARI: ChromaDB'deki '{CHROMA_COLLECTION_NAME}' koleksiyonu boş görünüyor. RAG sonuçları beklenildiği gibi olmayabilir.")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"ChromaDB'den '{CHROMA_COLLECTION_NAME}' koleksiyonu başarıyla yüklendi ve retriever oluşturuldu.")
    # Test sorgusu (isteğe bağlı)
    # test_docs = retriever.invoke("test")
    # print(f"ChromaDB test sorgusu sonucu {len(test_docs)} doküman döndürdü.")

except Exception as e:
    st.error(f"ChromaDB yüklenirken/erişilirken hata: {e}")
    st.stop()

# YENİ: NewsData.io API anahtarı kontrolü
if not NEWSDATA_API_KEY:
    st.warning("UYARI: NEWSDATA_API_KEY ortam değişkeni bulunamadı. "
               "Genel bilgi soruları için haber arama özelliği devre dışı kalacak "
               "ve sadece LLM'in genel bilgisi kullanılacaktır.")
    

# --- LangGraph Durum (State) Tanımı ---
class AgentState(TypedDict):
    question: str                 # Kullanıcının orijinal sorusu
    classification: str | None    # Sorunun sınıfı: "resmi_gazete", "general", "irrelevant"
    context: List[str] | None     # RAG için ChromaDB'den alınan içerikler
    answer: str | None            # Üretilen nihai cevap
    source: str | None            # Cevabın kaynağı ("Resmi Gazete", "Genel Bilgi", "Yanıt Yok")
    error: str | None             # İşlem sırasında oluşan hata mesajı


# --- LangGraph Düğümleri (Nodes) ---

# 1. Supervisor: Soruyu Sınıflandırma Düğümü (GÜNCELLENDİ)
def classify_question_node(state: AgentState):
    """Soruyu LLM kullanarak sınıflandırır ve cevaptan anahtar kelimeyi çıkarır."""
    print("--- Supervisor: Soruyu Sınıflandırıyor ---")
    question = state["question"]
    # Prompt'u biraz değiştirerek sınıflandırma kelimesini sonda belirtmesini teşvik edelim
    prompt = f"""Aşağıdaki kullanıcı sorusunu analiz et ve hangi kategoriye girdiğini belirle. Analizini kısaca yaptıktan sonra cevabının SONUNDA mutlaka şu kelimelerden birini KULLAN: 'resmi_gazete', 'general' veya 'irrelevant'.

    Kategoriler:
    - 'resmi_gazete': Türkiye Cumhuriyeti Resmi Gazetesi (kanun, yönetmelik, ihale, kiralama, satış, KHK, kararname, tebliğ, atama, ilan vb.) ile ilgili sorular.
    - 'general': Güncel olaylar, genel kültür, tanımlar, kişiler, yerler veya Resmi Gazete dışındaki diğer konular.
    - 'irrelevant': Anlamsız, saldırgan, tamamlanmamış veya cevaplanması mümkün olmayan sorular.

    Örnek Cevap Formatı:
    Soru: "2024 yılı bütçe kanunu ne zaman yayınlandı?"
    Analiz ve Kategori: Bu soru bir kanunla ilgili ve Resmi Gazete'de yayınlanır. resmi_gazete

    Soru: "Türkiye'nin başkenti neresi?"
    Analiz ve Kategori: Bu genel kültür sorusudur. general

    Soru: "mavi uyur mu?"
    Analiz ve Kategori: Bu soru anlamsızdır. irrelevant

    Kullanıcı Sorusu: "{question}"

    Analiz ve Kategori:"""

    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip().lower() # Yanıtı al ve küçük harfe çevir
        print(f"LLM Ham Yanıtı (Sınıflandırma): {response_text}")

        valid_classifications = ["resmi_gazete", "general", "irrelevant"]
        found_classification = None
        # Yanıtın sonundan başlayarak anahtar kelimeleri ara (en son bulunanı al)
        # Bunu yapmak için her kelimenin *son* geçtiği indeksi bulup en büyüğünü seçeceğiz
        max_index = -1

        for keyword in valid_classifications:
            index = response_text.rfind(keyword) # Kelimenin *son* bulunduğu indeksi verir
            if index > max_index: # Eğer bu keyword daha sonra (daha büyük indexte) bulunduysa
                max_index = index
                found_classification = keyword

        if found_classification:
            print(f"Çıkarılan Sınıflandırma (Son bulunan): {found_classification}")
            classification = found_classification
        else:
            # Eğer hiçbir anahtar kelime bulunamazsa, varsayılan olarak 'general' kullan
            # ve bir uyarı logla. Bu durum, LLM'in prompt'a hiç uymadığını gösterir.
            print(f"UYARI: LLM yanıtında geçerli sınıflandırma kelimesi ('resmi_gazete', 'general', 'irrelevant') bulunamadı. Yanıt: '{response_text}'. 'general' olarak kabul ediliyor.")
            classification = "general" # Güvenli varsayılan

        return {"classification": classification, "error": None}

    except Exception as e:
        print(f"HATA: Soru sınıflandırma sırasında: {e}")
        # Hata durumunda da güvenli bir varsayılan belirle
        return {"classification": "general", "error": f"Sınıflandırma sırasında LLM hatası: {e}"}


# 2. Resmi Gazete RAG Agent Düğümü
def resmi_gazete_rag_node(state: AgentState):
    """ChromaDB'den ilgili belgeleri alır ve LLM ile cevap üretir."""
    print("--- Agent: Resmi Gazete RAG ---")
    question = state["question"]
    try:
        print("ChromaDB'den dokümanlar alınıyor...")
        # retriever'ı doğrudan invoke et
        docs = retriever.invoke(question)
        context_list = [doc.page_content for doc in docs]
        context_str = "\n\n---\n\n".join(context_list) # String formatında context
        print(f"{len(docs)} doküman bulundu.")

        if not docs:
            print("Resmi Gazete için ilgili doküman bulunamadı.")
            context_str = "Resmi Gazete arşivinde bu konuyla doğrudan ilgili bir belge bulunamadı. Bu bilgiye dayanarak cevap ver."

        # Prompt'u biraz daha netleştirelim
        prompt = f"""Sen Türkiye Cumhuriyeti Resmi Gazetesi içerikleri konusunda uzman bir yapay zeka asistanısın.
        Aşağıda sana sağlanan Bağlam (Context) bölümündeki Resmi Gazete alıntılarını ve Kullanıcı Sorusu'nu dikkatlice incele.
        SADECE sağlanan Bağlam'daki bilgileri kullanarak Kullanıcı Sorusu'nu DOĞRUDAN ve NET bir şekilde cevapla.
        Eğer Bağlam soruyu cevaplamak için yeterli bilgi içermiyorsa, "Sağlanan Resmi Gazete belgelerinde bu soruya doğrudan cevap verecek bilgi bulunmamaktadır." şeklinde belirt.
        Kesinlikle Bağlam dışı bilgi kullanma veya yorum yapma.

        Bağlam (Context):
        ---
        {context_str}
        ---

        Kullanıcı Sorusu: {question}

        Cevap (Sadece Bağlama Göre):"""

        print("LLM ile cevap üretiliyor...")
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Cevabı state'e eklerken context'i de liste olarak ekle
        return {"context": context_list, "answer": answer, "source": "Resmi Gazete", "error": None}

    except Exception as e:
        print(f"HATA: Resmi Gazete RAG sırasında: {e}")
        return {"answer": "Resmi Gazete verilerine erişirken veya cevap oluştururken teknik bir sorun oluştu.", "source": "Hata", "error": str(e)}


# 3. Genel Bilgi Agent Düğümü
def general_knowledge_node(state: AgentState):
    """
    Kullanıcının sorusuyla ilgili NewsData.io'dan güncel haberleri arar,
    bulunan haberleri bağlam olarak kullanarak LLM ile cevap üretir.
    API anahtarı yoksa veya hata alınırsa sadece LLM kullanılır (fallback).
    """
    print("--- Agent: Genel Bilgi (NewsData.io ile) ---")
    question = state["question"]
    news_context_str = "Güncel haberler aranmadı veya bulunamadı." # Başlangıç değeri
    source = "Genel Bilgi (LLM)" # Başlangıç kaynağı

    if not NEWSDATA_API_KEY:
        print("UYARI: NewsData.io API anahtarı ayarlanmamış. Sadece LLM kullanılacak.")
        # API anahtarı yoksa doğrudan LLM'e git (fallback)
    else:
        try:
            print(f"NewsData.io API'si ile '{question}' sorgusu yapılıyor...")
            # API istemcisini başlat
            api = NewsDataApiClient(apikey=NEWSDATA_API_KEY)

            # Türkçe haberleri ara, en fazla 5 sonuç getir
            response = api.news_api(q=question, language="tr", size=5) # size değerini ayarlayabilirsin

            if response.get("status") == "success":
                articles = response.get("results", [])
                total_results = response.get("totalResults", 0)
                print(f"NewsData.io API'den {len(articles)}/{total_results} makale bulundu.")

                if articles:
                    # Bulunan makalelerden bir bağlam oluştur
                    context_parts = []
                    for i, article in enumerate(articles):
                        title = article.get('title', 'Başlık Yok')
                        description = article.get('description', 'Açıklama Yok')
                        pubDate = article.get('pubDate', 'Tarih Yok')
                        link = article.get('link', '#')
                        source_id = article.get('source_id', 'Kaynak Yok')
                        context_parts.append(f"Haber {i+1} ({source_id} - {pubDate}):\nBaşlık: {title}\nAçıklama: {description}\nLink: {link}")

                    news_context_str = "\n\n---\n\n".join(context_parts)
                    source = "Genel Bilgi (NewsData.io)" # Kaynağı güncelle
                    print("Haber içerikleri LLM için hazırlandı.")
                else:
                    news_context_str = "Bu konuyla ilgili güncel haber bulunamadı."
                    print("NewsData.io'da ilgili haber bulunamadı.")
            else:
                # API hatası durumunda logla ve LLM fallback yap
                error_msg = response.get("results", {}).get("message", "Bilinmeyen API hatası")
                print(f"HATA: NewsData.io API hatası: {error_msg}")
                news_context_str = f"Güncel haberler aranırken bir API hatası oluştu: {error_msg}"
                source = "Genel Bilgi (LLM - Haber API Hatası)"

        except Exception as e:
            # Genel API veya kütüphane hatası
            print(f"HATA: NewsData.io API çağrısı sırasında beklenmedik hata: {e}")
            news_context_str = f"Güncel haberler aranırken bir sistem hatası oluştu: {e}"
            source = "Genel Bilgi (LLM - Haber Sistemi Hatası)"

    # --- LLM ile Cevap Üretme ---
    print(f"Kaynak: {source}. LLM ile cevap üretiliyor...")
    # LLM'e verilecek prompt'u haber bağlamına göre ayarla
    prompt = f"""Sen güncel olaylar ve genel konularda bilgi veren bir asistansın.
Aşağıda kullanıcı sorusuyla ilgili olabilecek güncel haber özetleri bulunmaktadır (eğer varsa).
Bu haber özetlerini ve kendi genel bilgini kullanarak kullanıcı sorusunu cevapla.
Eğer haberler soruyu doğrudan yanıtlamıyorsa veya haber bulunamadıysa, bunu belirt ve soruyu genel bilginle cevaplamaya çalış.

Güncel Haber Özeti Bağlamı:
---
{news_context_str}
---

Kullanıcı Sorusu: {question}

Cevap:"""

    try:
        response_llm = llm.invoke(prompt)
        answer = response_llm.content.strip()
        # Genel bilgi node'u için context'i None yapalım, çünkü bu RAG context'i değil
        return {"context": None, "answer": answer, "source": source, "error": None}
    except Exception as e:
        print(f"HATA: Genel bilgi LLM çağrısı sırasında: {e}")
        # LLM hatasında bile bir cevap döndürmeye çalışalım
        fallback_answer = "Sorunuzu yanıtlarken bir sorunla karşılaştım. Lütfen daha sonra tekrar deneyin."
        if "API hatası" in news_context_str or "sistem hatası" in news_context_str:
            # Eğer API hatası varsa, bunu yanıta ekleyebiliriz
             fallback_answer = f"Güncel haberleri alırken bir sorun oluştuğu için sorunuzu yanıtlayamıyorum: {news_context_str}"

        return {"context": None, "answer": fallback_answer, "source": "Hata", "error": str(e)}


# 4. Fallback Agent Düğümü
def fallback_node(state: AgentState):
    """Uygun olmayan veya cevaplanamayan sorular için standart yanıt verir."""
    print("--- Agent: Fallback ---")
    answer = "Üzgünüm, bu soruya şu an için yanıt veremiyorum. Sorunuz anlaşılamamış veya bilgi alanımın dışında olabilir."
    # Fallback'te context olmaz
    return {"context": None, "answer": answer, "source": "Yanıt Yok", "error": None}


# --- LangGraph Yönlendirme Mantığı (Conditional Edges) ---
def route_question(state: AgentState):
    """Sınıflandırmaya göre bir sonraki düğümü belirler."""
    classification = state.get("classification")
    print(f"Yönlendirme Kararı: '{classification}' sınıflandırmasına göre yapılıyor.")

    if classification == "resmi_gazete":
        return "resmi_gazete_agent"
    elif classification == "general":
        return "general_agent"
    elif classification == "irrelevant":
        return "fallback_agent"
    else:
        # Bu durumun aslında classify_question_node'daki varsayılan atama ile
        # engellenmesi lazım ama yine de bir güvenlik önlemi olarak kalsın.
        print(f"UYARI: Geçersiz veya eksik sınıflandırma '{classification}'. Fallback'e yönlendiriliyor.")
        return "fallback_agent"


# --- LangGraph Grafiğini Oluşturma ---
workflow = StateGraph(AgentState)

# Düğümleri ekle
workflow.add_node("supervisor", classify_question_node)
workflow.add_node("resmi_gazete_agent", resmi_gazete_rag_node)
workflow.add_node("general_agent", general_knowledge_node)
workflow.add_node("fallback_agent", fallback_node)

# Giriş noktasını belirle
workflow.set_entry_point("supervisor")

# Koşullu kenarları ekle (Supervisor'dan sonra nereye gidilecek?)
workflow.add_conditional_edges(
    "supervisor",
    route_question,
    {
        # Hedef düğüm isimleri add_node ile tanımlananlarla eşleşmeli
        "resmi_gazete_agent": "resmi_gazete_agent",
        "general_agent": "general_agent",
        "fallback_agent": "fallback_agent",
    }
)

# Agent düğümlerinden sonra bitişe git (END)
workflow.add_edge("resmi_gazete_agent", END)
workflow.add_edge("general_agent", END)
workflow.add_edge("fallback_agent", END)

# Grafiği derle
try:
    app = workflow.compile()
    print("LangGraph grafiği başarıyla derlendi.")
except Exception as e:
    st.error(f"LangGraph grafiği derlenirken hata: {e}")
    st.stop()

# --- Streamlit Arayüzü ---
st.set_page_config(page_title="Haberbot", layout="wide")

st.title("Haberbot / Haber Arama")
st.caption(f"LLM: {OLLAMA_LLM} | Embeddings: {OLLAMA_EMBED_MODEL} | ChromaDB: {CHROMA_DB_PATH}")

# Chat geçmişini session state'de tut
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Kaynak ve context bilgilerini sadece asistan mesajlarında göster
        if message["role"] == "assistant":
            details = []
            if "source" in message and message["source"]:
                details.append(f"Kaynak: {message['source']}")
            if "context" in message and message["context"] and message["context"] is not None:
                 # Context'i sadece Resmi Gazete için gösterelim
                 if message.get("source") == "Resmi Gazete":
                      details.append("RAG Context Mevcut") # Sadece mevcut olduğunu belirtelim

            if details:
                 st.caption(" | ".join(details))

            # Context içeriğini expander içinde göster (varsa)
            if "context" in message and message["context"] and message["context"] is not None and message.get("source") == "Resmi Gazete":
                 with st.expander("Detay: RAG Bağlamı (Context)"):
                      # Her bir context parçasını ayrı ayrı gösterelim
                      for i, ctx in enumerate(message["context"]):
                           st.text_area(f"Chunk {i+1}", ctx, height=150, disabled=True)


# Kullanıcıdan input al
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chatbot cevabı için LangGraph'ı çalıştır
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        response_source = ""
        response_context = None # None olarak başlat
        error_message = None
        details_for_caption = []

        try:
            with st.spinner("Düşünüyor ve kaynakları araştırıyor..."):
                # LangGraph'ı invoke et
                inputs = {"question": prompt}
                # Akışı izlemek için stream() kullanabiliriz ama invoke() daha basit
                final_state = app.invoke(inputs, {"recursion_limit": 5}) # Sonsuz döngüleri engellemek için limit

                # Nihai durumu kontrol et
                if final_state and isinstance(final_state, dict):
                    full_response_content = final_state.get("answer", "Bir hata oluştu, cevap alınamadı.")
                    response_source = final_state.get("source", "Bilinmiyor")
                    error_message = final_state.get("error") # Hata mesajını al

                    # RAG context'ini al (liste olarak)
                    # Sadece 'Resmi Gazete' kaynağından geliyorsa anlamlı
                    if response_source == "Resmi Gazete":
                        response_context = final_state.get("context") # None olabilir
                else:
                     full_response_content = "Beklenmedik bir durum oluştu, geçerli bir sonuç alınamadı."
                     response_source = "Hata"


            # Cevabı yazdır
            message_placeholder.markdown(full_response_content)

            # Caption için detayları oluştur
            if response_source:
                details_for_caption.append(f"Kaynak: {response_source}")
            if response_context is not None and response_source == "Resmi Gazete":
                 details_for_caption.append("RAG Context Mevcut")

            if details_for_caption:
                 st.caption(" | ".join(details_for_caption))

            # Hata varsa göster
            if error_message:
                 st.error(f"İşlem sırasında bir uyarı/hata oluştu: {error_message}")

            # Context'i expander içinde göster (varsa)
            if response_context is not None and response_source == "Resmi Gazete":
                 with st.expander("Detay: RAG Bağlamı (Context)"):
                      for i, ctx in enumerate(response_context):
                           st.text_area(f"Chunk {i+1}", ctx, height=150, disabled=True)


        except Exception as e:
            import traceback
            print(f"Streamlit Arayüz Hatası: {traceback.format_exc()}") # Konsola tam hatayı yazdır
            full_response_content = f"Üzgünüm, isteğinizi işlerken beklenmedik bir sistem hatası oluştu: {e}"
            response_source = "Sistem Hatası"
            message_placeholder.error(full_response_content)

    # Asistanın cevabını (context ile birlikte) geçmişe ekle
    assistant_message = {
        "role": "assistant",
        "content": full_response_content,
        "source": response_source,
        "context": response_context # Context'i her zaman ekle (None olsa bile)
    }
    st.session_state.messages.append(assistant_message)