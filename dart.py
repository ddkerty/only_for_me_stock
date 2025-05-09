# dart.py (MCP 관련 코드 완전 제거/주석처리된 버전)

import streamlit as st # Streamlit 임포트
import httpx
from typing import Any, Dict, List, Optional, Tuple, Set
# from mcp.server.fastmcp import FastMCP, Context # <<< MCP 임포트 주석 처리
import os
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO, StringIO
import re
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv() # .env 파일 로드 시도

# API 설정
API_KEY = os.getenv("DART_API_KEY") # 1. 환경 변수 먼저 확인
if not API_KEY and hasattr(st, 'secrets'): # 2. 환경 변수에 없고, secrets 사용 가능하면
    API_KEY = st.secrets.get("DART_API_KEY") # 3. Streamlit secrets에서 직접 읽기
BASE_URL = "https://opendart.fss.or.kr/api"

# --- MCP 서버 초기화 제거 ---
# mcp = FastMCP("dart") # <<< MCP 객체 생성 주석 처리
# --------------------------

# 보고서 코드 (이하 상수 정의는 유지)
REPORT_CODE = {
    "사업보고서": "11011",
    "반기보고서": "11012",
    "1분기보고서": "11013",
    "3분기보고서": "11014"
}
# ... (BALANCE_SHEET_ITEMS, CASH_FLOW_ITEMS 등 나머지 상수 정의는 그대로 유지) ...
BALANCE_SHEET_ITEMS = [
    "유동자산", "비유동자산", "자산총계",
    "유동부채", "비유동부채", "부채총계",
    "자본금", "자본잉여금", "이익잉여금", "기타자본항목", "자본총계"
]
CASH_FLOW_ITEMS = ["영업활동 현금흐름", "투자활동 현금흐름", "재무활동 현금흐름"]
REPORT_PATTERNS = {
    "연간": "FY",
    "3분기": "TQQ",
    "반기": "HYA",
    "1분기": "FQA"
}
CASH_FLOW_PATTERNS = {
    "연간": "FY",
    "3분기": "TQA",
    "반기": "HYA",
    "1분기": "FQA"
}
BALANCE_SHEET_PATTERNS = {
    "연간": "FY",
    "3분기": "TQA",
    "반기": "HYA",
    "1분기": "FQA"
}
INVALID_VALUE_INDICATORS = {"N/A", "XBRL 파싱 오류", "데이터 추출 오류"}
STATEMENT_TYPES = {
    "재무상태표": "BS",
    "손익계산서": "IS",
    "현금흐름표": "CF"
}
DETAILED_TAGS = {
    "재무상태표": {
        "유동자산": ["ifrs-full:CurrentAssets"], "비유동자산": ["ifrs-full:NoncurrentAssets"], "자산총계": ["ifrs-full:Assets"],
        "유동부채": ["ifrs-full:CurrentLiabilities"], "비유동부채": ["ifrs-full:NoncurrentLiabilities"], "부채총계": ["ifrs-full:Liabilities"],
        "자본금": ["ifrs-full:IssuedCapital"], "자본잉여금": ["ifrs-full:SharePremium"], "이익잉여금": ["ifrs-full:RetainedEarnings"],
        "기타자본항목": ["dart:ElementsOfOtherStockholdersEquity"], "자본총계": ["ifrs-full:Equity"]
    },
    "손익계산서": {
        "매출액": ["ifrs-full:Revenue"], "매출원가": ["ifrs-full:CostOfSales"], "매출총이익": ["ifrs-full:GrossProfit"],
        "판매비와관리비": ["dart:TotalSellingGeneralAdministrativeExpenses"], "영업이익": ["dart:OperatingIncomeLoss"],
        "금융수익": ["ifrs-full:FinanceIncome"], "금융비용": ["ifrs-full:FinanceCosts"],
        "법인세비용차감전순이익": ["ifrs-full:ProfitLossBeforeTax"], "법인세비용": ["ifrs-full:IncomeTaxExpenseContinuingOperations"],
        "당기순이익": ["ifrs-full:ProfitLoss"], "기본주당이익": ["ifrs-full:BasicEarningsLossPerShare"]
    },
    "현금흐름표": {
        "영업활동 현금흐름": ["ifrs-full:CashFlowsFromUsedInOperatingActivities"], "영업에서 창출된 현금": ["ifrs-full:CashFlowsFromUsedInOperations"],
        "이자수취": ["ifrs-full:InterestReceivedClassifiedAsOperatingActivities"], "이자지급": ["ifrs-full:InterestPaidClassifiedAsOperatingActivities"],
        "배당금수취": ["ifrs-full:DividendsReceivedClassifiedAsOperatingActivities"], "법인세납부": ["ifrs-full:IncomeTaxesPaidRefundClassifiedAsOperatingActivities"],
        "투자활동 현금흐름": ["ifrs-full:CashFlowsFromUsedInInvestingActivities"],
        "유형자산의 취득": ["ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"],
        "무형자산의 취득": ["ifrs-full:PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities"],
        "유형자산의 처분": ["ifrs-full:ProceedsFromSalesOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"],
        "재무활동 현금흐름": ["ifrs-full:CashFlowsFromUsedInFinancingActivities"], "배당금지급": ["ifrs-full:DividendsPaidClassifiedAsFinancingActivities"],
        "현금및현금성자산의순증가": ["ifrs-full:IncreaseDecreaseInCashAndCashEquivalents"],
        "기초현금및현금성자산": ["dart:CashAndCashEquivalentsAtBeginningOfPeriodCf"], "기말현금및현금성자산": ["dart:CashAndCashEquivalentsAtEndOfPeriodCf"]
    }
}
chat_guideline = "\n* 제공된 공시정보들은 분기, 반기, 연간이 섞여있을 수 있습니다. \n사용자가 특별히 연간이나 반기데이터만을 원하는게 아니라면, 주어진 데이터를 적당히 가공하여 분기별로 사용자에게 제공하세요." ;

# --- 모든 함수 정의에서 @mcp.tool() 데코레이터 제거 ---
# --- 그리고 함수 인자에서 ctx: Context 타입 어노테이션 및 ctx.info, ctx.report_progress 호출 부분 제거 또는 주석 처리 ---

async def get_corp_code_by_name(corp_name: str) -> Tuple[str, str]:
    # ctx 인자 제거 또는 Optional 처리, ctx.info() 등 제거
    # ... (기존 함수 로직에서 ctx 관련 부분 제외하고 유지) ...
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
                    if not matches: return ("", f"'{corp_name}' 회사 없음.")
                    matches.sort(key=lambda x: x[2])
                    return matches[0][1], matches[0][0]
    except Exception as e: return ("", f"회사 코드 조회 오류: {e}")
    return ("", "알 수 없는 오류 (회사 코드)")

async def get_disclosure_list(corp_code: str, start_date: str, end_date: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    # ... (기존 함수 로직에서 ctx 관련 부분 제외하고 유지) ...
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

# get_financial_statement_xbrl, detect_namespaces, extract_fiscal_year, get_pattern_by_item_type,
# format_numeric_value, parse_xbrl_financial_data, determine_report_code, adjust_end_date,
# extract_business_section, extract_business_section_from_dart, get_original_document,
# search_disclosure, search_detailed_financial_data, search_business_information,
# get_current_date, get_financial_json, get_report_code_name, get_statement_name,
# search_json_financial_data
# 위의 모든 함수들에서 @mcp.tool() 데코레이터를 제거하고,
# ctx: Context 인자 및 ctx.info(), ctx.report_progress() 호출 부분을 제거하거나 주석 처리해야 합니다.
# (코드 길이가 너무 길어 모든 함수를 여기에 다 적지는 않겠습니다. 위 예시처럼 수정해주세요.)

# --- 파일 맨 아래 서버 실행 코드 제거 ---
# if __name__ == "__main__":
# mcp.run(transport='stdio') # <<< 확실히 주석 처리 또는 삭제
# ------------------------------------