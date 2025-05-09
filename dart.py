# dart.py (MCP 관련 코드 완전 제거/주석처리 및 API 키 로드 수정)

import streamlit as st # Streamlit 임포트
import httpx
from typing import Any, Dict, List, Optional, Tuple, Set
# from mcp.server.fastmcp import FastMCP, Context # <<< 삭제 또는 주석 처리
import os
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO, StringIO
import re
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# API 설정
API_KEY = None # 초기값 None으로 설정
if hasattr(st, 'secrets') and "DART_API_KEY" in st.secrets: # 1. Streamlit secrets 먼저 확인
    API_KEY = st.secrets.get("DART_API_KEY")
    if API_KEY:
        print("INFO: DART API Key loaded from Streamlit Secrets.") # 로그 확인용
if not API_KEY: # Secrets에 없으면 환경 변수 확인
    API_KEY = os.getenv("DART_API_KEY")
    if API_KEY:
        print("INFO: DART API Key loaded from environment variable.") # 로그 확인용
if not API_KEY:
    print("CRITICAL WARNING: DART_API_KEY is not set. DART functionalities will fail.")
    # st.error("DART API 키가 설정되지 않았습니다.") # dart.py에서 직접 st.error 호출은 지양

BASE_URL = "https://opendart.fss.or.kr/api"

# --- MCP 서버 초기화 제거 ---
# mcp = FastMCP("dart") # <<< 삭제 또는 주석 처리
# --------------------------

# --- 상수 정의 (유지) ---
REPORT_CODE = {"사업보고서": "11011", "반기보고서": "11012", "1분기보고서": "11013", "3분기보고서": "11014"}
BALANCE_SHEET_ITEMS = ["유동자산","비유동자산","자산총계","유동부채","비유동부채","부채총계","자본금","자본잉여금","이익잉여금","기타자본항목","자본총계"]
CASH_FLOW_ITEMS = ["영업활동 현금흐름","투자활동 현금흐름","재무활동 현금흐름"]
REPORT_PATTERNS = {"연간":"FY","3분기":"TQQ","반기":"HYA","1분기":"FQA"}
CASH_FLOW_PATTERNS = {"연간":"FY","3분기":"TQA","반기":"HYA","1분기":"FQA"}
BALANCE_SHEET_PATTERNS = {"연간":"FY","3분기":"TQA","반기":"HYA","1분기":"FQA"}
INVALID_VALUE_INDICATORS = {"N/A","XBRL 파싱 오류","데이터 추출 오류"}
STATEMENT_TYPES = {"재무상태표":"BS","손익계산서":"IS","현금흐름표":"CF"}
DETAILED_TAGS = {
    "재무상태표": {"유동자산":["ifrs-full:CurrentAssets"],"비유동자산":["ifrs-full:NoncurrentAssets"],"자산총계":["ifrs-full:Assets"],"유동부채":["ifrs-full:CurrentLiabilities"],"비유동부채":["ifrs-full:NoncurrentLiabilities"],"부채총계":["ifrs-full:Liabilities"],"자본금":["ifrs-full:IssuedCapital"],"자본잉여금":["ifrs-full:SharePremium"],"이익잉여금":["ifrs-full:RetainedEarnings"],"기타자본항목":["dart:ElementsOfOtherStockholdersEquity"],"자본총계":["ifrs-full:Equity"]},
    "손익계산서": {"매출액":["ifrs-full:Revenue"],"매출원가":["ifrs-full:CostOfSales"],"매출총이익":["ifrs-full:GrossProfit"],"판매비와관리비":["dart:TotalSellingGeneralAdministrativeExpenses"],"영업이익":["dart:OperatingIncomeLoss"],"금융수익":["ifrs-full:FinanceIncome"],"금융비용":["ifrs-full:FinanceCosts"],"법인세비용차감전순이익":["ifrs-full:ProfitLossBeforeTax"],"법인세비용":["ifrs-full:IncomeTaxExpenseContinuingOperations"],"당기순이익":["ifrs-full:ProfitLoss"],"기본주당이익":["ifrs-full:BasicEarningsLossPerShare"]},
    "현금흐름표": {"영업활동 현금흐름":["ifrs-full:CashFlowsFromUsedInOperatingActivities"],"영업에서 창출된 현금":["ifrs-full:CashFlowsFromUsedInOperations"],"이자수취":["ifrs-full:InterestReceivedClassifiedAsOperatingActivities"],"이자지급":["ifrs-full:InterestPaidClassifiedAsOperatingActivities"],"배당금수취":["ifrs-full:DividendsReceivedClassifiedAsOperatingActivities"],"법인세납부":["ifrs-full:IncomeTaxesPaidRefundClassifiedAsOperatingActivities"],"투자활동 현금흐름":["ifrs-full:CashFlowsFromUsedInInvestingActivities"],"유형자산의 취득":["ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"],"무형자산의 취득":["ifrs-full:PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities"],"유형자산의 처분":["ifrs-full:ProceedsFromSalesOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"],"재무활동 현금흐름":["ifrs-full:CashFlowsFromUsedInFinancingActivities"],"배당금지급":["ifrs-full:DividendsPaidClassifiedAsFinancingActivities"],"현금및현금성자산의순증가":["ifrs-full:IncreaseDecreaseInCashAndCashEquivalents"],"기초현금및현금성자산":["dart:CashAndCashEquivalentsAtBeginningOfPeriodCf"],"기말현금및현금성자산":["dart:CashAndCashEquivalentsAtEndOfPeriodCf"]}
}
chat_guideline = "\n* 제공된 공시정보들은 분기, 반기, 연간이 섞여있을 수 있습니다. \n사용자가 특별히 연간이나 반기데이터만을 원하는게 아니라면, 주어진 데이터를 적당히 가공하여 분기별로 사용자에게 제공하세요."

# --- 모든 함수 정의에서 @mcp.tool() 데코레이터 및 ctx 인자 제거 ---
# (이전 답변에서 제시한 대로, 모든 함수의 @mcp.tool() 데코레이터와 ctx 관련 코드를 제거해야 합니다.)
# (코드 길이 관계상 모든 함수를 나열하지 않고, 예시만 남깁니다.)

async def get_corp_code_by_name(corp_name: str) -> Tuple[str, str]:
    # ctx 인자 제거, 내부 ctx.info() 등 제거
    if not API_KEY: return ("", "DART API 키가 설정되지 않았습니다.") # API 키 확인
    url = f"{BASE_URL}/corpCode.xml?crtfc_key={API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200: return ("", f"API 요청 실패: HTTP {response.status_code}")
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                with zf.open('CORPCODE.xml') as xml_file:
                    tree = ET.parse(xml_file); root = tree.getroot()
                    matches = []
                    for company in root.findall('.//list'):
                        name = company.findtext('corp_name'); stock_code = company.findtext('stock_code')
                        if not (stock_code and stock_code.strip()): continue
                        if name and corp_name in name:
                            score = abs(len(name) - len(corp_name)) + (10 if not name.startswith(corp_name) else 0)
                            matches.append((name, company.findtext('corp_code'), score))
                    if not matches: return ("", f"'{corp_name}' 회사를 찾을 수 없습니다.")
                    matches.sort(key=lambda x: x[2])
                    return matches[0][1], matches[0][0]
    except Exception as e: return ("", f"회사 코드 조회 오류: {e}")
    return ("", "알 수 없는 오류 (회사 코드)")

async def get_disclosure_list(corp_code: str, start_date: str, end_date: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not API_KEY: return [], "DART API 키가 설정되지 않았습니다."
    url = f"{BASE_URL}/list.json?crtfc_key={API_KEY}&corp_code={corp_code}&bgn_de={start_date}&end_de={end_date}&pblntf_ty=A&page_count=100"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200: return [], f"API 요청 실패: HTTP {response.status_code}"
            result = response.json()
            if result.get('status') != '000': return [], f"DART API 오류: {result.get('status')} - {result.get('message')}"
            return result.get('list', []), None
    except Exception as e: return [], f"공시 목록 조회 오류: {e}"
    return [], "알 수 없는 오류 (공시 목록)"

async def extract_business_section_from_dart(rcept_no: str, section_type: str) -> str:
    if not API_KEY: return "DART API 키가 설정되지 않았습니다."
    document_text, binary_data = await get_original_document(rcept_no)
    if binary_data is None: return f"공시서류 다운로드 실패: {document_text}"
    # extract_business_section 함수는 이 파일 내에 정의되어 있어야 함
    return extract_business_section(document_text, section_type)

# ... (get_financial_statement_xbrl, detect_namespaces, extract_fiscal_year 등 나머지 함수들도 MCP 관련 코드 제거 및 필요시 API_KEY 확인 로직 추가) ...
# ... (모든 @mcp.tool() 데코레이터 제거, ctx 인자 및 관련 코드 제거) ...

# 예시: extract_business_section 함수 (MCP 관련 코드 없음)
def extract_business_section(document_text: str, section_type: str) -> str:
    import re
    section_patterns = {
        '사업의 개요': r'<TITLE[^>]*>(?:\d+\.\s*)?사업의\s*개요[^<]*</TITLE>(.*?)(?=<TITLE|</SECTION)',
        # ... (다른 패턴들) ...
    }
    if section_type not in section_patterns: return f"지원하지 않는 섹션 유형입니다."
    pattern = section_patterns[section_type]
    matches = re.search(pattern, document_text, re.DOTALL | re.IGNORECASE)
    if not matches: return f"'{section_type}' 섹션을 찾을 수 없습니다."
    section_text = matches.group(1)
    clean_text = re.sub(r'<[^>]*>', ' ', section_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

async def get_original_document(rcept_no: str) -> Tuple[str, Optional[bytes]]:
    if not API_KEY: return "DART API 키가 설정되지 않았습니다.", None
    url = f"{BASE_URL}/document.xml?crtfc_key={API_KEY}&rcept_no={rcept_no}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200: return f"API 요청 실패: HTTP {response.status_code}", None
            try:
                root = ET.fromstring(response.content)
                status = root.findtext('status'); message = root.findtext('message')
                if status and message: return f"DART API 오류: {status} - {message}", None
            except ET.ParseError: pass # 정상적인 ZIP 파일일 수 있음
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                file_list = zf.namelist()
                if not file_list: return "ZIP 파일 내 파일 없음.", None
                target_file = min(file_list, key=len) # 가장 짧은 이름의 파일 선택
                with zf.open(target_file) as doc_file:
                    file_content = doc_file.read()
                    encodings = ['utf-8', 'euc-kr', 'cp949']
                    for enc in encodings:
                        try: return file_content.decode(enc), file_content
                        except UnicodeDecodeError: continue
                    return "파일 인코딩 변환 실패.", file_content
    except Exception as e: return f"원본 문서 다운로드 오류: {e}", None
    return "알 수 없는 오류 (원본 문서)", None


# --- 파일 맨 아래 서버 실행 코드 제거 ---
# if __name__ == "__main__":
# mcp.run(transport='stdio')
# ------------------------------------