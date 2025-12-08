import streamlit as st

st.markdown("""
    <style>
    /* 1. تصغير المسافات الرأسية داخل كل رسالة */
    .stChatMessage {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* 2. (اختياري) تصغير حجم الأيقونة/الصورة الجانبية */
    .stChatMessage .stAvatar {
        width: 32px !important;
        height: 32px !important;
    }
    
    /* 3. (اختياري) تصغير حجم الخط قليلاً */
    .stChatMessage p {
        font-size: 0.95rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# تجربة للتأكد من الشكل
with st.chat_message("user"):
    st.write("هل الحجم كدة مناسب؟")

with st.chat_message("assistant"):
    st.write("نعم، المسافات قلت والحجم أصبح ألطف.")
