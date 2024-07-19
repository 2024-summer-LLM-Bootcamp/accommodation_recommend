import streamlit as st

# 페이지 구성
st.set_page_config(layout="wide")

# 섹션을 위한 데이터 예시
sections = {
    "강릉 초당순두부": "JMTGR",
    "가평펜션": "바베큐파티",
    "국민대": "노잼",
    "구디": "구로디지털단지엔 먹을 게 없다",
    "Jeju": "제주도 가고 싶다",
    "반얀트리 호텔": "비싸서 못감",
    "풀파티": "!!!",
    "홍대 게스트하우스": "게하",
    "오사카": "매우 더운 방",
    "보스턴": "동부"
}

# 상태를 저장할 수 있는 상태 변수
if 'num_sections' not in st.session_state:
    st.session_state.num_sections = 3  # 초기에 표시할 섹션 수

# 검색 창과 초기화 버튼
with st.form(key='search_form', border=False):
    search_query = st.text_input('검색', label_visibility="hidden")
    submit_button = st.form_submit_button(label='검색')

if submit_button:
    # st.write(f'You searched for: {search_query}')
    filtered_sections = {k: v for k, v in sections.items() if search_query.lower() in v.lower() or search_query.lower() in k.lower()}
else:
    filtered_sections = sections


# 섹션을 담는 컨테이너
sections_to_display = list(filtered_sections.items())[:st.session_state.num_sections]

for section, content in sections_to_display:
    with st.container(border=True):
        st.subheader(section)
        st.write(content)
        
# 더 보기 버튼
if st.session_state.num_sections < len(filtered_sections):
    if st.button('더보기'):
        st.session_state.num_sections += 3
        
# 초기화 버튼
if st.button('초기화'):
    st.session_state.num_sections = 3
    st.experimental_rerun()