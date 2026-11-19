import os
import json
import joblib
import pandas as pd

from groq import Groq
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST


# Koneksi ke Groq API
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# Memuat model Machine Learning
MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "ml_model",
    "model_kos.joblib"
)

model = joblib.load(MODEL_PATH)


# Menampilkan dashboard dan daftar kos
def dashboard(request):
    dataset_path = os.path.join(
        settings.BASE_DIR,
        "dataset",
        "kos_madiun.csv"
    )
    df = pd.read_csv(dataset_path)

    daftar_kos = []

    for _, row in df.iterrows():
        harga = float(row["harga"])

        if harga <= 800:
            kategori = "Murah"
        elif harga <= 1200:
            kategori = "Menengah"
        else:
            kategori = "Mahal"

        harga_rupiah = round(harga * 1000)

        daftar_kos.append({
            "nama": row["nama"],
            "harga_display": f"{harga_rupiah:,}".replace(",", "."),
            "jarak_km": float(row["jarak_km"]),
            "ac": str(row["ac"]).lower(),
            "rating": float(row["rating"]),
            "kategori": kategori
        })

    rata_rata_harga = round(float(df["harga"].mean()) * 1000)
    rata_rata_rating = round(float(df["rating"].mean()), 2)

    return render(request, "prediksi/dashboard.html", {
        "daftar_kos": daftar_kos,
        "total_kos": len(df),
        "rata_rata_harga_display": (
            f"{rata_rata_harga:,}".replace(",", ".")
        ),
        "rata_rata_rating": rata_rata_rating
    })


# API untuk prediksi harga kos
@require_POST
def api_prediksi(request):
    try:
        data = json.loads(request.body)

        jarak = float(data.get("jarak_km"))
        rating = float(data.get("rating"))
        ac = data.get("ac")

        # Validasi input
        if not 0.6 <= jarak <= 5.0:
            return JsonResponse({
                "status": "error",
                "pesan": "Jarak harus antara 0.6 sampai 5 km."
            }, status=400)

        if not 3.1 <= rating <= 4.9:
            return JsonResponse({
                "status": "error",
                "pesan": "Rating harus antara 3.1 sampai 4.9."
            }, status=400)

        if ac not in ["ya", "tidak"]:
            return JsonResponse({
                "status": "error",
                "pesan": "Pilihan AC tidak valid."
            }, status=400)

        # Mengubah fasilitas AC menjadi angka
        ac_value = 1 if ac == "ya" else 0

        # Menyiapkan data untuk model
        input_data = pd.DataFrame([{
            "jarak_km": jarak,
            "ac": ac_value,
            "rating": rating
        }])

        # Prediksi model masih dalam satuan ribuan Rupiah
        hasil = model.predict(input_data)[0]

        return JsonResponse({
            "status": "success",
            "prediksi_harga": round(float(hasil), 2),
            "input": {
                "jarak_km": jarak,
                "rating": rating,
                "ac": ac
            },
            "pesan": "Prediksi harga kos berhasil dilakukan."
        })

    except (ValueError, TypeError, json.JSONDecodeError):
        return JsonResponse({
            "status": "error",
            "pesan": "Input tidak valid. Periksa kembali data yang dimasukkan."
        }, status=400)


# API Grafik dengan penjelasan AI dari Groq
def api_grafik(request):
    try:
        dataset_path = os.path.join(
            settings.BASE_DIR,
            "dataset",
            "kos_madiun.csv"
        )

        df = pd.read_csv(dataset_path)

        # Mengelompokkan data berdasarkan jarak
        df["kelompok_jarak"] = pd.cut(
            df["jarak_km"],
            bins=[0, 1, 2, 3, 4, 6],
            labels=["0-1 km", "1-2 km", "2-3 km", "3-4 km", "4-5 km"],
            include_lowest=True
        )

        grafik_jarak = (
            df.groupby("kelompok_jarak", observed=False)["harga"]
            .mean()
            .round(2)
        )

        grafik_ac = df.groupby("ac")["harga"].mean()

        harga_ac = round(float(grafik_ac.get("ya", 0)), 2)
        harga_tanpa_ac = round(float(grafik_ac.get("tidak", 0)), 2)

        # Mengubah data harga menjadi Rupiah untuk grafik dan AI
        data_analisis = {
            "jumlah_data": len(df),
            "rata_rata_harga_rupiah": round(
                float(df["harga"].mean()) * 1000
            ),
            "kelompok_jarak": [
                {
                    "jarak": str(label),
                    "rata_rata_harga_rupiah": (
                        round(float(nilai) * 1000)
                        if pd.notna(nilai) else 0
                    )
                }
                for label, nilai in grafik_jarak.items()
            ],
            "harga_dengan_ac_rupiah": round(harga_ac * 1000),
            "harga_tanpa_ac_rupiah": round(harga_tanpa_ac * 1000)
        }

        # Meminta Groq menganalisis data grafik
        try:
            response = groq_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Kamu adalah asisten analis data untuk "
                            "aplikasi prediksi harga kos. "
                            "Jelaskan data menggunakan bahasa Indonesia "
                            "yang sederhana, jelas, dan mudah dipahami. "
                            "Gunakan hanya data yang diberikan. "
                            "Jangan mengarang angka atau menyimpulkan "
                            "hubungan sebab-akibat. Semua nilai harga "
                            "sudah dalam Rupiah, bukan ribuan Rupiah."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Analisis data grafik harga kos berikut:\n"
                            + json.dumps(
                                data_analisis,
                                ensure_ascii=False
                            )
                            + """

Berikan hasil dalam format JSON yang valid dengan dua properti:
1. "jarak": penjelasan pola rata-rata harga kos berdasarkan kelompok jarak.
2. "ac": penjelasan perbandingan rata-rata harga kos dengan AC dan tanpa AC.

Masing-masing penjelasan terdiri dari 2-3 kalimat.
Tulis nominal dalam Rupiah dengan format Indonesia, misalnya Rp1.175.000.
Jangan gunakan markdown atau teks di luar JSON.
"""
                        )
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500
            )

            hasil_ai = json.loads(
                response.choices[0].message.content
            )

            penjelasan = {
                "jarak": hasil_ai["jarak"],
                "ac": hasil_ai["ac"]
            }

        except Exception:
            penjelasan = {
                "jarak": (
                    "Penjelasan AI belum tersedia. "
                    "Silakan muat ulang halaman."
                ),
                "ac": (
                    "Penjelasan AI belum tersedia. "
                    "Silakan muat ulang halaman."
                )
            }

        # Mengirim data grafik dan penjelasan AI ke dashboard
        return JsonResponse({
            "status": "success",
            "grafik_jarak": {
                "labels": [
                    str(x) for x in grafik_jarak.index
                ],
                "data": [
                    round(float(x) * 1000) if pd.notna(x) else 0
                    for x in grafik_jarak.values
                ]
            },
            "grafik_ac": {
                "labels": ["Dengan AC", "Tanpa AC"],
                "data": [
                    round(harga_ac * 1000),
                    round(harga_tanpa_ac * 1000)
                ]
            },
            "penjelasan": penjelasan
        })

    except Exception as e:
        return JsonResponse({
            "status": "error",
            "pesan": str(e)
        }, status=500)


# Fungsi untuk menguji koneksi Groq
def test_groq(request):
    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": "Jawab singkat: Koneksi Groq berhasil."
                }
            ],
            temperature=0.3,
            max_tokens=100
        )

        return JsonResponse({
            "status": "success",
            "jawaban": response.choices[0].message.content
        })

    except Exception as e:
        return JsonResponse({
            "status": "error",
            "pesan": str(e)
        }, status=500)