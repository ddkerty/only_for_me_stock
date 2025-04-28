# ============== 📊 종합 분석 탭 ==============
if page == "📊 종합 분석":
    st.title("📊 종합 분석 결과")
    st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공.")
    st.markdown("---")
    analyze_button_main_disabled = not comprehensive_analysis_possible
    if analyze_button_main_disabled: st.error("API 키 로드 실패. 종합 분석 불가.")
    analyze_button_main = st.button("🚀 종합 분석 시작!", use_container_width=True, type="primary", key="analyze_main_button", disabled=analyze_button_main_disabled)
    results_placeholder = st.container()

    if analyze_button_main:
        ticker = st.session_state.get('main_ticker', "AAPL")
        years = st.session_state.get('analysis_years', 2)
        days = st.session_state.get('forecast_days', 30)
        periods = st.session_state.get('num_trend_periods', 4)
        cp_prior = st.session_state.get('changepoint_prior', 0.05)
        avg_p = st.session_state.get('avg_price', 0.0)
        qty = st.session_state.get('quantity', 0)

        if not ticker:
            results_placeholder.warning("종목 티커 입력 필요.")
        else:
            ticker_proc = ticker.strip().upper()
            with st.spinner(f"{ticker_proc} 종합 분석 중..."): # <-- 분석 시작 스피너
                try:
                    # --- run_cached_analysis 한 번만 호출 ---
                    results = run_cached_analysis(ticker_proc, NEWS_API_KEY, FRED_API_KEY, years, days, periods, cp_prior)

                    # --- 상세 결과 표시 로직 통합 (중복 제거) ---
                    # --- 결과 처리 시작 (오류 또는 정상) ---
                    if results and isinstance(results, dict):
                        if "error" in results:
                            results_placeholder.error(f"분석 실패: {results['error']}")
                        else:
                            # 분석 성공 시 결과 표시
                            results_placeholder.empty() # 이전 메시지 비우기

                            # --- MAPE 경고 배너 삽입 ---
                            if results.get("warn_high_mape"):
                                m = results.get("mape", 0.0)
                                results_placeholder.warning(
                                    f"🔴 모델 정확도 낮음 (MAPE {m:.1f}%). 예측 신뢰도에 주의하세요!"
                                )
                            # ------------------------------

                            st.header(f"📈 {ticker_proc} 분석 결과 (민감도: {cp_prior:.3f})")

                            # 1. 요약 정보
                            st.subheader("요약 정보")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("현재가", f"${results.get('current_price', 'N/A')}")
                            col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                            col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))

                            # 2. 기본적 분석
                            st.subheader("📊 기업 기본 정보")
                            fundamentals = results.get('fundamentals')
                            if fundamentals and isinstance(fundamentals, dict) and fundamentals.get("시가총액", "N/A") != "N/A":
                                colf1, colf2, colf3 = st.columns(3)
                                with colf1:
                                    st.metric("시가총액", fundamentals.get("시가총액", "N/A"))
                                    st.metric("PER", fundamentals.get("PER", "N/A"))
                                with colf2:
                                    st.metric("EPS", fundamentals.get("EPS", "N/A"))
                                    st.metric("Beta", fundamentals.get("베타", "N/A"))
                                with colf3:
                                    st.metric("배당수익률", fundamentals.get("배당수익률", "N/A"))
                                    st.metric("업종", fundamentals.get("업종", "N/A"))
                                industry = fundamentals.get("산업", "N/A")
                                summary = fundamentals.get("요약", "N/A")
                                if industry != "N/A": st.markdown(f"**산업:** {industry}")
                                if summary != "N/A":
                                    with st.expander("회사 요약 보기"):
                                        st.write(summary)
                                st.caption("Data Source: Yahoo Finance")
                            else: st.warning("기업 기본 정보 로드 실패.")
                            st.divider()

                            # 3. 주요 재무 추세
                            st.subheader(f"📈 주요 재무 추세 (최근 {periods} 분기)")
                            tab_titles = ["영업이익률(%)", "ROE(%)", "부채비율", "유동비율"]
                            tabs = st.tabs(tab_titles)
                            trend_data_map = {
                                "영업이익률(%)": ('operating_margin_trend', 'Op Margin (%)', "{:.2f}%"),
                                "ROE(%)": ('roe_trend', 'ROE (%)', "{:.2f}%"),
                                "부채비율": ('debt_to_equity_trend', 'D/E Ratio', "{:.2f}"),
                                "유동비율": ('current_ratio_trend', 'Current Ratio', "{:.2f}")
                            }
                            for i, title in enumerate(tab_titles):
                                with tabs[i]:
                                    data_key, col_name, style_format = trend_data_map[title]
                                    trend_data = results.get(data_key)
                                    if trend_data and isinstance(trend_data, list) and len(trend_data) > 0:
                                        try:
                                            df_trend = pd.DataFrame(trend_data)
                                            df_trend['Date'] = pd.to_datetime(df_trend['Date'])
                                            df_trend.set_index('Date', inplace=True)
                                            if col_name in df_trend.columns:
                                                st.line_chart(df_trend[[col_name]])
                                                with st.expander("데이터 보기"):
                                                    st.dataframe(df_trend[[col_name]].style.format({col_name: style_format}), use_container_width=True)
                                            else:
                                                st.error(f"'{col_name}' 컬럼 없음.")
                                        except Exception as e:
                                            st.error(f"{title} 표시 오류: {e}")
                                    else:
                                        st.info(f"{title} 추세 데이터 없음.")
                            st.divider()

                            # 4. 기술적 분석 차트 (종합)
                            st.subheader("기술적 분석 차트 (종합)")
                            stock_chart_fig = results.get('stock_chart_fig')
                            if stock_chart_fig:
                                st.plotly_chart(stock_chart_fig, use_container_width=True)
                            else:
                                st.warning("주가 차트 생성 실패 (종합).")
                            st.divider()

                            # 5. 시장 심리 분석
                            st.subheader("시장 심리 분석")
                            col_news, col_fng = st.columns([2, 1])
                            with col_news:
                                st.markdown("**📰 뉴스 감정 분석**")
                                news_sentiment = results.get('news_sentiment', ["정보 없음."])
                                if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                                    st.info(news_sentiment[0])
                                    if len(news_sentiment) > 1:
                                        with st.expander("뉴스 목록 보기"):
                                            for line in news_sentiment[1:]:
                                                st.write(f"- {line}")
                                else:
                                    st.write(str(news_sentiment))
                            with col_fng:
                                st.markdown("**😨 공포-탐욕 지수**")
                                fng_index = results.get('fear_greed_index', "N/A")
                                if isinstance(fng_index, dict):
                                    st.metric("현재 지수", fng_index.get('value', 'N/A'), fng_index.get('classification', ''))
                                else:
                                    st.write(fng_index)
                            st.divider() # 뉴스/F&G 섹션 후 구분선

                            # 6. Prophet 주가 예측
                            st.subheader("Prophet 주가 예측")
                            forecast_fig = results.get('forecast_fig')
                            forecast_data_list = results.get('prophet_forecast')
                            if forecast_fig:
                                st.plotly_chart(forecast_fig, use_container_width=True)
                            elif isinstance(forecast_data_list, str): # 예측 불가 시 메시지
                                st.info(forecast_data_list)
                            else:
                                st.warning("예측 차트 생성 실패.")

                            if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                                st.markdown("**📊 예측 데이터 (최근 10일)**")
                                try:
                                    df_fcst = pd.DataFrame(forecast_data_list)
                                    df_fcst['ds'] = pd.to_datetime(df_fcst['ds'])
                                    df_fcst_display = df_fcst.sort_values("ds").iloc[-10:].copy()
                                    df_fcst_display['ds'] = df_fcst_display['ds'].dt.strftime('%Y-%m-%d')
                                    st.dataframe(
                                        df_fcst_display[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].style.format({
                                            'yhat': "{:.2f}", 'yhat_lower': "{:.2f}", 'yhat_upper': "{:.2f}"
                                        }),
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.error(f"예측 데이터 표시 오류: {e}")

                            cv_plot_path = results.get('cv_plot_path')
                            if cv_plot_path and os.path.exists(cv_plot_path):
                                st.markdown("**📉 교차 검증 결과 (MAPE)**")
                                st.image(cv_plot_path, caption="MAPE (낮을수록 정확)")
                            elif cv_plot_path is None and isinstance(forecast_data_list, list):
                                st.caption("교차 검증(CV) 결과 없음.")
                            st.divider()

                            # 7. 리스크 트래커
                            st.subheader("🚨 리스크 트래커 (예측 기반)")
                            risk_days, max_loss_pct, max_loss_amt = 0, 0, 0 # 초기화
                            if avg_p > 0 and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                                try:
                                    df_pred = pd.DataFrame(forecast_data_list)
                                    required_fcst_cols = ['ds', 'yhat_lower']
                                    if not all(col in df_pred.columns for col in required_fcst_cols):
                                        st.warning("리스크 분석 위한 예측 컬럼 부족.")
                                    else:
                                        df_pred['ds'] = pd.to_datetime(df_pred['ds'])
                                        df_pred['yhat_lower'] = pd.to_numeric(df_pred['yhat_lower'], 'coerce')
                                        df_pred.dropna(subset=['yhat_lower'], inplace=True)

                                        if not df_pred.empty:
                                            df_pred['평단가'] = avg_p
                                            df_pred['리스크 여부'] = df_pred['yhat_lower'] < avg_p
                                            df_pred['예상 손실률'] = np.where(df_pred['리스크 여부'], ((df_pred['yhat_lower'] - avg_p) / avg_p) * 100, 0)
                                            if qty > 0:
                                                df_pred['예상 손실액'] = np.where(df_pred['리스크 여부'], (df_pred['yhat_lower'] - avg_p) * qty, 0)
                                            else:
                                                df_pred['예상 손실액'] = 0

                                            risk_days = df_pred['리스크 여부'].sum()
                                            if risk_days > 0:
                                                max_loss_pct = df_pred['예상 손실률'].min()
                                                max_loss_amt = df_pred['예상 손실액'].min() if qty > 0 else 0

                                            st.markdown("##### 리스크 요약")
                                            col_r1, col_r2, col_r3 = st.columns(3)
                                            col_r1.metric("⚠️ < 평단가 일수", f"{risk_days}일 / {days}일")
                                            col_r2.metric("📉 Max 손실률", f"{max_loss_pct:.2f}%")
                                            if qty > 0: col_r3.metric("💸 Max 손실액", f"${max_loss_amt:,.2f}")

                                            if risk_days > 0: st.warning(f"{days}일 예측 중 **{risk_days}일** 평단가(${avg_p:.2f}) 하회 가능성.")
                                            else: st.success(f"{days}일간 평단가(${avg_p:.2f}) 하회 가능성 낮음.")

                                            st.markdown("##### 평단가 vs 예측 구간 비교")
                                            fig_risk = go.Figure()
                                            # ... (기존 리스크 차트 그리는 로직 동일) ...
                                            if 'yhat_upper' in df_pred.columns:
                                                fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat_upper'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper'))
                                            fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat_lower'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower', fill='tonexty', fillcolor='rgba(0,100,80,0.1)'))
                                            if 'yhat' in df_pred.columns:
                                                fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat'], mode='lines', line=dict(dash='dash', color='rgba(0,100,80,0.6)'), name='Forecast'))
                                            fig_risk.add_hline(y=avg_p, line_dash="dot", line_color="red", annotation_text=f"평단가: ${avg_p:.2f}", annotation_position="bottom right")
                                            df_risk_periods = df_pred[df_pred['리스크 여부']]
                                            if not df_risk_periods.empty:
                                                fig_risk.add_trace(go.Scatter(x=df_risk_periods['ds'], y=df_risk_periods['yhat_lower'], mode='markers', marker_symbol='x', marker_color='red', name='Risk Day'))
                                            fig_risk.update_layout(hovermode="x unified")
                                            st.plotly_chart(fig_risk, use_container_width=True)

                                            if risk_days > 0:
                                                with st.expander(f"리스크 예측일 상세 데이터 ({risk_days}일)"):
                                                    df_risk_days_display = df_pred[df_pred['리스크 여부']].copy()
                                                    df_risk_days_display['ds'] = df_risk_days_display['ds'].dt.strftime('%Y-%m-%d')
                                                    cols_show = ['ds', 'yhat_lower', '평단가', '예상 손실률']
                                                    if qty > 0: cols_show.append('예상 손실액')
                                                    st.dataframe(df_risk_days_display[cols_show].style.format({"yhat_lower":"{:.2f}", "평단가":"{:.2f}", "예상 손실률":"{:.2f}%", "예상 손실액":"${:,.2f}"}), use_container_width=True)
                                        else:
                                            st.info("리스크 분석 위한 유효한 예측 하한선 데이터 없음.")
                                except Exception as risk_calc_err:
                                    st.error(f"리스크 트래커 오류: {risk_calc_err}")
                                    logging.error(f"Risk tracker error: {traceback.format_exc()}")
                            elif avg_p <= 0:
                                st.info("⬅️ '평단가' 입력 시 리스크 분석 결과 확인 가능.")
                            else: # 예측 데이터 자체가 없는 경우
                                st.warning("예측 데이터 유효하지 않아 리스크 분석 불가.")
                            st.divider()

                            # 8. 자동 분석 결과 요약 (summary_points 최종 한 번만 출력)
                            st.subheader("🧐 자동 분석 결과 요약 (참고용)")
                            summary_points = []

                            # 예측 요약 추가
                            if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                                try:
                                    # ... (기존 예측 요약 계산 로직 동일) ...
                                    df_pred = pd.DataFrame(forecast_data_list).sort_values("ds")
                                    start_pred = df_pred["yhat"].iloc[0]
                                    end_pred   = df_pred["yhat"].iloc[-1]
                                    trend_obs = ("상승" if end_pred > start_pred * 1.02 else "하락" if end_pred < start_pred * 0.98 else "횡보")
                                    lower = df_pred["yhat_lower"].min()
                                    upper = df_pred["yhat_upper"].max()
                                    summary_points.append(f"- **예측:** 향후 {days}일간 **{trend_obs}** 추세 (${lower:.2f} ~ ${upper:.2f})")
                                except Exception as e:
                                    logging.warning(f"예측 요약 생성 오류: {e}")
                                    summary_points.append("- 예측: 요약 생성 오류")

                            # 뉴스 요약 추가
                            news_res = results.get('news_sentiment')
                            if isinstance(news_res, list) and len(news_res) > 0 and ":" in news_res[0]:
                                try:
                                    score_part = news_res[0].split(":")[-1].strip()
                                    avg_score = float(score_part)
                                    sentiment_desc = "긍정적" if avg_score > 0.05 else "부정적" if avg_score < -0.05 else "중립적"
                                    summary_points.append(f"- **뉴스:** 평균 감성 {avg_score:.2f}, **{sentiment_desc}** 분위기.")
                                except Exception as e:
                                    logging.warning(f"뉴스 요약 오류: {e}")
                                    summary_points.append("- 뉴스: 요약 오류.")

                            # F&G 요약 추가
                            fng_res = results.get('fear_greed_index')
                            if isinstance(fng_res, dict):
                                summary_points.append(f"- **시장 심리:** 공포-탐욕 {fng_res.get('value', 'N/A')} (**{fng_res.get('classification', 'N/A')}**).")

                            # 기본 정보 요약 추가
                            if fundamentals and isinstance(fundamentals, dict):
                                per = fundamentals.get("PER", "N/A")
                                sector = fundamentals.get("업종", "N/A")
                                parts = []
                                if per != "N/A": parts.append(f"PER {per}")
                                if sector != "N/A": parts.append(f"업종 '{sector}'")
                                if parts: summary_points.append(f"- **기본 정보:** {', '.join(parts)}.")

                            # 재무 추세 요약 추가
                            trend_parts = []
                            try:
                                op_margin_trend = results.get('operating_margin_trend')
                                roe_trend = results.get('roe_trend')
                                debt_trend = results.get('debt_to_equity_trend')
                                current_trend = results.get('current_ratio_trend')
                                # 마지막 분기 데이터 사용
                                if op_margin_trend and op_margin_trend[-1]: trend_parts.append(f"영업익률 {op_margin_trend[-1].get('Op Margin (%)', 'N/A'):.2f}%")
                                if roe_trend and roe_trend[-1]: trend_parts.append(f"ROE {roe_trend[-1].get('ROE (%)', 'N/A'):.2f}%")
                                if debt_trend and debt_trend[-1]: trend_parts.append(f"부채비율 {debt_trend[-1].get('D/E Ratio', 'N/A'):.2f}")
                                if current_trend and current_trend[-1]: trend_parts.append(f"유동비율 {current_trend[-1].get('Current Ratio', 'N/A'):.2f}")
                                if trend_parts: summary_points.append(f"- **최근 재무:** {', '.join(trend_parts)}.")
                            except Exception as e:
                                logging.warning(f"재무 추세 요약 오류: {e}")
                                summary_points.append("- 최근 재무: 요약 오류.")

                            # 리스크 요약 추가
                            if avg_p > 0 and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                                if risk_days > 0:
                                    summary_points.append(f"- **리스크:** {days}일 중 **{risk_days}일** 평단가 하회 가능성 (Max 손실률: **{max_loss_pct:.2f}%**).")
                                else:
                                    summary_points.append(f"- **리스크:** 평단가(${avg_p:.2f}) 하회 가능성 낮음.")
                            elif avg_p > 0: # 평단가는 입력했으나 예측 데이터가 없는 경우
                                summary_points.append("- 리스크: 예측 불가로 분석 제한됨.")

                            # --- summary_points 최종 한 번만 출력 ---
                            if summary_points:
                                st.markdown("\n".join(summary_points))
                                st.caption("⚠️ **주의:** 자동 생성된 요약이며 투자 조언이 아닙니다.")
                            else:
                                st.write("분석 요약 생성 불가.")
                            # --- 상세 결과 표시 끝 ---

                    elif results is None: # results 자체가 None인 경우
                         results_placeholder.error("분석 결과 처리 중 예상치 못한 오류 발생 (결과 없음).")
                    else: # dict 형태가 아닌 경우 등 기타 문제
                        results_placeholder.error("분석 결과 처리 중 오류 발생 (형식 오류).")
                    # --- 결과 처리 끝 ---

                except Exception as e:
                    error_traceback = traceback.format_exc()
                    logging.error(f"종합 분석 실행 오류: {e}\n{error_traceback}")
                    results_placeholder.error(f"앱 실행 중 오류 발생: {e}")
                    # st.exception(e) # 디버깅 시 유용

    else: # 버튼 클릭 전
        if comprehensive_analysis_possible:
            results_placeholder.info("⬅️ 사이드바 설정 후 '종합 분석 시작' 버튼 클릭.")

# --- 이하 기술 분석 탭 코드는 동일 ---
# ... (elif page == "📈 기술 분석": 부분) ...