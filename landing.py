def landing_page():
    st.title("ANALISIS SAHAM DENGAN REGRESI BERGANDA IDX 10")

    # Penambahan gambar (opsional)
    st.image('path_to_your_image.jpg', width=700) # Sesuaikan path dan ukuran sesuai kebutuhan

    st.markdown("""
    <div style="text-align: justify;">
        Platform ini dirancang untuk memberikan analisis tentang sepuluh saham dengan kapitalisasi terbesar di Indonesia (IDX 10). 
        Kami menggunakan data historis terkini untuk menyediakan berbagai insight yang menarik tentang saham tersebut.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Fitur Utama:")
    st.markdown("""
    - **Grafik Time Series**: Memvisualisasikan pergerakan harga saham sepanjang waktu.
    - **Grafik Volume Perdagangan**: Menampilkan volume perdagangan saham.
    - **Grafik Scatter**: Analisis hubungan antar variabel.
    - **Grafik Candlestick**: Detail pergerakan harga dengan candlestick chart.
    - **Heatmap Korelasi**: Menjelajahi korelasi antar atribut saham.
    - **Prediksi Harga Saham**: Menggunakan model regresi untuk prediksi harga.
    """)