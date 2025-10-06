# ===================================
# TAB 6: Segmentation
# ===================================
with tab4:
    st.header("👥 Customer Segmentation")
    try:
        seg = ch_df("""
            SELECT cs.rfm_segment, cs.rfm_score,
                   cf.txn_amount_30d, cf.txn_count_30d, cf.p2p_ratio_30d, cf.days_since_last_txn
            FROM customer_segments cs
            LEFT JOIN card_features cf ON cs.hpan = cf.hpan
        """, ["rfm_segment", "rfm_score", "txn_amount_30d", "txn_count_30d", "p2p_ratio_30d", "days_since_last_txn"])
        if seg.empty:
            st.info("Нет customer_segments/card_features. Запустите пайплайн сегментации.")
        else:
            summary = seg.groupby("rfm_segment").agg(
                clients=("rfm_score", "count"),
                avg_amount=("txn_amount_30d", "mean"),
                avg_txn=("txn_count_30d", "mean"),
                avg_recency=("days_since_last_txn", "mean"),
                avg_p2p=("p2p_ratio_30d", "mean")
            ).reset_index()
            summary["share"] = (summary["clients"] / summary["clients"].sum() * 100)
            c1, c2 = st.columns([2, 3])
            with c1:
                st.subheader("Распределение RFM")
                st.plotly_chart(px.pie(summary, values="clients", names="rfm_segment", hole=0.45),
                                use_container_width=True)
            with c2:
                st.subheader("Метрики по сегментам")
                show = summary[
                    ["rfm_segment", "clients", "share", "avg_amount", "avg_txn", "avg_recency", "avg_p2p"]].round(
                    2).rename(columns={
                    "rfm_segment": "Сегмент", "clients": "Клиентов", "share": "Доля,%", "avg_amount": "Avg объём",
                    "avg_txn": "Avg частота", "avg_recency": "Avg Recency (дн.)", "avg_p2p": "Avg P2P"
                })
                st.dataframe(show, use_container_width=True)
                st.download_button("⬇️ CSV (метрики)", data=show.to_csv(index=False).encode("utf-8"),
                                   file_name="segments_metrics.csv")

            st.subheader("R×F heatmap (из RFM_score)")
            tmp = seg.copy()
            tmp["R"] = tmp["rfm_score"].str[0].astype(str)
            tmp["F"] = tmp["rfm_score"].str[1].astype(str)
            hmp = tmp.pivot_table(index="R", columns="F", values="rfm_score", aggfunc="count").fillna(0)
            st.plotly_chart(px.imshow(hmp, text_auto=True, color_continuous_scale="Blues"), use_container_width=True)

            st.subheader("Рекомендации")
            st.markdown("""
- 🏆 Champions — удержание, персональные VIP-предложения  
- 💎 Loyal — кросс/апсейл, программы лояльности  
- 🌱 Potential — welcome-кампании  
- ⚠️ At Risk — реактивация  
- 😴 Hibernating — win-back  
- ❌ Lost — агрессивный win-back или исключение
""")
    except Exception as e:
        st.error(f"Segmentation: {e}")

# ===================================
# TAB 7: Enhanced Forecasting
# ===================================
with tab5:
    st.header("📊 Enhanced Forecasting")
    st.caption(f"Прогнозирование с метриками качества и диагностикой")

    # Forecast mode selector
    forecast_mode = st.radio(
        "Режим прогнозирования",
        ["📈 Простой прогноз", "🔬 Диагностика рядов", "📊 Валидация моделей", "🎯 Сценарный анализ"],
        horizontal=True
    )

    if forecast_mode == "📈 Простой прогноз":
        st.subheader("Настройки прогноза")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_to_forecast = st.selectbox(
                "Метрика",
                ["volume", "transactions", "unique_cards"],
                format_func=lambda x: {
                    "volume": "Объём (UZS)",
                    "transactions": "Количество транзакций",
                    "unique_cards": "Уникальные карты"
                }[x]
            )
        with col2:
            forecast_horizon = st.number_input("Горизонт (дней)", 7, 90, 14)
        with col3:
            holdout_days = st.number_input("Holdout (дней)", 0, 30, 14)
        with col4:
            confidence_level = st.selectbox("Доверительный интервал", [80, 90, 95], index=2)

        if st.button("🚀 Построить прогноз", type="primary"):
            if HAS_ENHANCED_FORECAST:
                try:
                    with st.spinner("Загрузка данных..."):
                        # Get historical data
                        if metric_to_forecast == "volume":
                            query = f"""
                            SELECT 
                                toDate(transaction_date) AS date,
                                sum(amount_uzs) AS value
                            FROM {selected_table}
                            WHERE transaction_date >= today() - INTERVAL 120 DAY
                            GROUP BY date
                            ORDER BY date
                            """
                        elif metric_to_forecast == "transactions":
                            query = f"""
                            SELECT 
                                toDate(transaction_date) AS date,
                                count() AS value
                            FROM {selected_table}
                            WHERE transaction_date >= today() - INTERVAL 120 DAY
                            GROUP BY date
                            ORDER BY date
                            """
                        else:  # unique_cards
                            if has_col(selected_table, "hpan"):
                                query = f"""
                                SELECT 
                                    toDate(transaction_date) AS date,
                                    uniq(hpan) AS value
                                FROM {selected_table}
                                WHERE transaction_date >= today() - INTERVAL 120 DAY
                                GROUP BY date
                                ORDER BY date
                                """
                            else:
                                st.error("Нет колонки hpan для подсчета уникальных карт")
                                query = None

                        if query:
                            df = ch_df(query)
                            df.columns = ["date", metric_to_forecast]
                        else:
                            df = pd.DataFrame()

                    if not df.empty:
                        # Initialize forecaster
                        forecaster = EnhancedForecaster()

                        # Prepare data
                        train, holdout = forecaster.prepare_data(
                            df,
                            value_col=metric_to_forecast,
                            holdout_days=holdout_days
                        )

                        with st.spinner("Генерация прогнозов..."):
                            # Generate multiple forecasts
                            forecasts = []
                            forecast_names = []

                            # Seasonal Naive
                            naive_fc = forecaster.seasonal_naive_forecast(
                                train, forecast_horizon, value_col=metric_to_forecast
                            )
                            forecasts.append(naive_fc)
                            forecast_names.append("Seasonal Naive")

                            # Prophet if available
                            prophet_fc = forecaster.prophet_forecast(
                                train, forecast_horizon, value_col=metric_to_forecast
                            )
                            if prophet_fc is not None:
                                forecasts.append(prophet_fc)
                                forecast_names.append("Prophet")

                            # Ensemble
                            ensemble = forecaster.ensemble_forecast(forecasts)

                        # Evaluate on holdout if available
                        metrics_dict = {}
                        if len(holdout) > 0:
                            st.info(f"📊 Оценка на holdout периоде ({holdout_days} дней)")

                            cols = st.columns(len(forecasts) + 1)

                            for i, (fc, name) in enumerate(zip(forecasts, forecast_names)):
                                metrics = forecaster.evaluate_forecast(
                                    fc, holdout, train, value_col=metric_to_forecast
                                )
                                metrics_dict[name] = metrics

                                with cols[i]:
                                    st.markdown(f"**{name}**")
                                    if "mape" in metrics:
                                        st.metric("MAPE", f"{metrics['mape']:.1f}%")
                                    if "coverage_95" in metrics:
                                        st.metric("Coverage 95%", f"{metrics['coverage_95']:.0f}%")

                            # Ensemble metrics
                            ensemble_metrics = forecaster.evaluate_forecast(
                                ensemble, holdout, train, value_col=metric_to_forecast
                            )
                            metrics_dict["Ensemble"] = ensemble_metrics

                            with cols[-1]:
                                st.markdown("**🎯 Ensemble**")
                                if "mape" in ensemble_metrics:
                                    st.metric("MAPE", f"{ensemble_metrics['mape']:.1f}%")
                                if "coverage_95" in ensemble_metrics:
                                    st.metric("Coverage 95%", f"{ensemble_metrics['coverage_95']:.0f}%")

                        # Visualization
                        st.subheader("📈 Визуализация прогноза")

                        fig = go.Figure()

                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=train["date"],
                            y=train[metric_to_forecast],
                            mode="lines",
                            name="История",
                            line=dict(color="black", width=2)
                        ))

                        # Holdout if available
                        if len(holdout) > 0:
                            fig.add_trace(go.Scatter(
                                x=holdout["date"],
                                y=holdout[metric_to_forecast],
                                mode="markers",
                                name="Holdout (факт)",
                                marker=dict(color="red", size=8)
                            ))

                        # Forecasts
                        colors = ["blue", "green", "purple", "orange"]
                        for fc, name, color in zip(forecasts, forecast_names, colors):
                            fig.add_trace(go.Scatter(
                                x=fc["date"],
                                y=fc["yhat"],
                                mode="lines",
                                name=f"{name} прогноз",
                                line=dict(color=color, width=1, dash="dot")
                            ))

                        # Ensemble with confidence interval
                        fig.add_trace(go.Scatter(
                            x=ensemble["date"],
                            y=ensemble["yhat_upper"],
                            mode="lines",
                            name="Upper bound",
                            line=dict(width=0),
                            showlegend=False
                        ))

                        fig.add_trace(go.Scatter(
                            x=ensemble["date"],
                            y=ensemble["yhat_lower"],
                            mode="lines",
                            name="Lower bound",
                            line=dict(width=0),
                            fillcolor="rgba(68, 68, 68, 0.2)",
                            fill="tonexty",
                            showlegend=False
                        ))

                        fig.add_trace(go.Scatter(
                            x=ensemble["date"],
                            y=ensemble["yhat"],
                            mode="lines+markers",
                            name="Ensemble прогноз",
                            line=dict(color="red", width=2),
                            marker=dict(size=6)
                        ))

                        fig.update_layout(
                            title=f"Прогноз: {metric_to_forecast}",
                            xaxis_title="Дата",
                            yaxis_title={
                                "volume": "Объём (UZS)",
                                "transactions": "Количество",
                                "unique_cards": "Карты"
                            }[metric_to_forecast],
                            height=500,
                            hovermode="x unified"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Export results
                        with st.expander("💾 Экспорт результатов"):
                            # Prepare export data
                            export_df = ensemble.copy()
                            export_df["metric"] = metric_to_forecast
                            export_df["model"] = "ensemble"

                            # Add metrics if available
                            if metrics_dict:
                                st.json(metrics_dict)

                            # Download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = export_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Скачать прогноз (CSV)",
                                    data=csv,
                                    file_name=f"forecast_{metric_to_forecast}_{datetime.now():%Y%m%d}.csv",
                                    mime="text/csv"
                                )

                            with col2:
                                if metrics_dict:
                                    metrics_json = json.dumps(metrics_dict, indent=2)
                                    st.download_button(
                                        "📊 Скачать метрики (JSON)",
                                        data=metrics_json,
                                        file_name=f"metrics_{metric_to_forecast}_{datetime.now():%Y%m%d}.json",
                                        mime="application/json"
                                    )
                    else:
                        st.warning("Нет данных для прогнозирования")

                except Exception as e:
                    st.error(f"Ошибка прогнозирования: {e}")
            else:
                st.error("Enhanced Forecasting модуль не установлен. Выполните: pip install statsmodels prophet")

    elif forecast_mode == "🔬 Диагностика рядов":
        st.subheader("Диагностика временных рядов")

        if HAS_ENHANCED_FORECAST:
            try:
                # Load time series
                query = f"""
                SELECT 
                    toDate(transaction_date) AS date,
                    sum(amount_uzs) AS volume,
                    count() AS transactions
                FROM {selected_table}
                WHERE transaction_date >= today() - INTERVAL 90 DAY
                GROUP BY date
                ORDER BY date
                """
                df = ch_df(query)

                if not df.empty:
                    df.columns = ["date", "volume", "transactions"]

                    # Select series to diagnose
                    series_name = st.selectbox("Выберите ряд", ["volume", "transactions"])

                    # Initialize diagnostics
                    diagnostics = TimeSeriesDiagnostics()

                    # STL decomposition
                    with st.spinner("Выполняю STL декомпозицию..."):
                        ts = pd.Series(
                            df[series_name].values,
                            index=pd.DatetimeIndex(df["date"])
                        )

                        stl_result = diagnostics.stl_decompose(ts)

                    if stl_result:
                        # Display decomposition
                        st.subheader("📊 STL Декомпозиция")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Сила тренда",
                                f"{stl_result.get('strength_trend', 0):.2f}",
                                help="Близко к 1 = сильный тренд"
                            )
                        with col2:
                            st.metric(
                                "Сила сезонности",
                                f"{stl_result.get('strength_seasonal', 0):.2f}",
                                help="Близко к 1 = сильная сезонность"
                            )

                        # Plot components
                        fig = go.Figure()

                        components = ["trend", "seasonal", "residual"]
                        for i, comp in enumerate(components):
                            if comp in stl_result:
                                fig.add_trace(go.Scatter(
                                    x=df["date"],
                                    y=stl_result[comp],
                                    mode="lines",
                                    name=comp.capitalize()
                                ))

                        fig.update_layout(
                            title="Компоненты временного ряда",
                            xaxis_title="Дата",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Anomaly detection
                    st.subheader("🔍 Обнаружение аномалий")

                    anomalies = diagnostics.detect_anomalies(ts, threshold=2.5)
                    anomaly_dates = df.loc[anomalies, "date"].tolist()

                    if anomaly_dates:
                        st.warning(f"Обнаружено {len(anomaly_dates)} аномалий")

                        # Plot with anomalies
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=df["date"],
                            y=df[series_name],
                            mode="lines",
                            name="Данные"
                        ))

                        fig.add_trace(go.Scatter(
                            x=df.loc[anomalies, "date"],
                            y=df.loc[anomalies, series_name],
                            mode="markers",
                            name="Аномалии",
                            marker=dict(color="red", size=10)
                        ))

                        fig.update_layout(
                            title=f"Аномалии в {series_name}",
                            xaxis_title="Дата",
                            height=350
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # List anomaly dates
                        with st.expander("Даты аномалий"):
                            for d in anomaly_dates:
                                st.write(f"• {d}")
                    else:
                        st.success("Аномалии не обнаружены")

                    # Autocorrelation test
                    st.subheader("📈 Тест автокорреляции")

                    # Calculate residuals (simple detrending)
                    residuals = ts - ts.rolling(window=7, center=True).mean()
                    residuals = residuals.dropna()

                    ljung_result = diagnostics.ljung_box_test(residuals, lags=14)

                    if ljung_result:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Ljung-Box статистика",
                                f"{ljung_result.get('statistic', 0):.2f}"
                            )
                        with col2:
                            st.metric(
                                "P-value",
                                f"{ljung_result.get('p_value', 1):.4f}",
                                help="< 0.05 означает наличие автокорреляции"
                            )

                        if ljung_result.get("autocorrelated"):
                            st.warning("⚠️ Обнаружена автокорреляция в остатках")
                        else:
                            st.success("✅ Автокорреляция не обнаружена")

            except Exception as e:
                st.error(f"Ошибка диагностики: {e}")
        else:
            st.error("Enhanced Forecasting модуль не установлен")

    elif forecast_mode == "📊 Валидация моделей":
        st.subheader("Rolling Origin Cross-Validation")

        if HAS_ENHANCED_FORECAST:
            col1, col2, col3 = st.columns(3)
            with col1:
                n_splits = st.number_input("Количество фолдов", 2, 10, 3)
            with col2:
                test_size = st.number_input("Размер теста (дней)", 7, 30, 14)
            with col3:
                metric_name = st.selectbox("Метрика", ["volume", "transactions"])

            if st.button("🔄 Запустить валидацию", type="primary"):
                try:
                    with st.spinner("Загрузка данных..."):
                        query = f"""
                        SELECT 
                            toDate(transaction_date) AS date,
                            {"sum(amount_uzs)" if metric_name == "volume" else "count()"} AS value
                        FROM {selected_table}
                        WHERE transaction_date >= today() - INTERVAL 180 DAY
                        GROUP BY date
                        ORDER BY date
                        """
                        df = ch_df(query)
                        df.columns = ["date", metric_name]

                    if not df.empty:
                        # Initialize forecaster
                        forecaster = EnhancedForecaster()

                        with st.spinner(f"Валидация на {n_splits} фолдах..."):
                            cv_results = forecaster.rolling_origin_validation(
                                df,
                                n_splits=n_splits,
                                test_size=test_size,
                                value_col=metric_name
                            )

                        if not cv_results.empty:
                            # Summary statistics
                            st.subheader("📊 Результаты кросс-валидации")

                            summary = cv_results.groupby("model")[["mape", "smape", "mase", "rmse"]].agg(
                                ["mean", "std"])

                            # Display metrics
                            for model in summary.index:
                                with st.expander(f"Модель: {model}"):
                                    col1, col2, col3, col4 = st.columns(4)

                                    with col1:
                                        mean_mape = summary.loc[model, ("mape", "mean")]
                                        std_mape = summary.loc[model, ("mape", "std")]
                                        st.metric(
                                            "MAPE",
                                            f"{mean_mape:.1f}%",
                                            f"±{std_mape:.1f}%"
                                        )

                                    with col2:
                                        mean_smape = summary.loc[model, ("smape", "mean")]
                                        std_smape = summary.loc[model, ("smape", "std")]
                                        st.metric(
                                            "sMAPE",
                                            f"{mean_smape:.1f}%",
                                            f"±{std_smape:.1f}%"
                                        )

                                    with col3:
                                        mean_mase = summary.loc[model, ("mase", "mean")]
                                        std_mase = summary.loc[model, ("mase", "std")]
                                        st.metric(
                                            "MASE",
                                            f"{mean_mase:.2f}",
                                            f"±{std_mase:.2f}"
                                        )

                                    with col4:
                                        mean_rmse = summary.loc[model, ("rmse", "mean")]
                                        st.metric(
                                            "RMSE",
                                            f"{mean_rmse:,.0f}"
                                        )

                            # Box plot of metrics
                            fig = go.Figure()

                            for model in cv_results["model"].unique():
                                model_data = cv_results[cv_results["model"] == model]
                                fig.add_trace(go.Box(
                                    y=model_data["mape"],
                                    name=model,
                                    boxmean=True
                                ))

                            fig.update_layout(
                                title="Распределение MAPE по фолдам",
                                yaxis_title="MAPE (%)",
                                height=400
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Download results
                            csv = cv_results.to_csv(index=False)
                            st.download_button(
                                "📥 Скачать результаты валидации",
                                data=csv,
                                file_name=f"cv_results_{metric_name}_{datetime.now():%Y%m%d}.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"Ошибка валидации: {e}")
        else:
            st.error("Enhanced Forecasting модуль не установлен")

    else:  # Сценарный анализ
        st.subheader("🎯 Сценарный анализ")
        st.info("Моделирование влияния изменений на прогноз")

        # Scenario parameters
        col1, col2 = st.columns(2)
        with col1:
            scenario_type = st.selectbox(
                "Тип сценария",
                ["Изменение тренда", "Сезонный шок", "Разовое событие"]
            )
        with col2:
            impact_percent = st.slider(
                "Величина воздействия (%)",
                -50, 50, 0, step=5
            )

        if scenario_type == "Изменение тренда":
            st.markdown("""
            **Изменение тренда** - постепенное изменение базового уровня
            - Положительное значение = рост
            - Отрицательное значение = спад
            """)
        elif scenario_type == "Сезонный шок":
            st.markdown("""
            **Сезонный шок** - изменение сезонной компоненты
            - Влияет на недельную периодичность
            - Может моделировать праздники/акции
            """)
        else:
            event_date = st.date_input("Дата события")
            event_duration = st.slider("Длительность (дней)", 1, 14, 3)

        if st.button("📊 Построить сценарий"):
            st.info("Сценарный анализ будет добавлен в следующей версии")
            # TODO: Implement scenario analysis

# ===================================
# TAB 8: Hierarchical Forecast
# ===================================
with tabHier:
    st.header("🏗 Hierarchical Forecast (из файлов)")
    merch_file = Path(find_artifact("forecast_hier_merchant.csv"))
    mcc_file = Path(find_artifact("forecast_hier_mcc.csv"))
    total_file = Path(find_artifact("forecast_hier_total.csv"))
    if total_file.exists():
        df_total = pd.read_csv(total_file, parse_dates=["ds"])
        st.plotly_chart(px.line(df_total, x="ds", y="yhat", title="TOTAL (Bottom-Up)"), use_container_width=True)
    if mcc_file.exists():
        df_mcc = pd.read_csv(mcc_file, parse_dates=["ds"])
        mccs = sorted(df_mcc["mcc"].unique().tolist())
        sl = st.selectbox("MCC", mccs, index=0)
        st.plotly_chart(px.line(df_mcc[df_mcc["mcc"] == sl], x="ds", y="yhat", title=f"MCC {sl} (BU)"),
                        use_container_width=True)

# ===================================
# TAB 13: NL Assistant
# ===================================
with tab8:
    st.header("😊 NL-ассистент (Claude)");
    st.caption(f"Источник по умолчанию: {selected_table}")
    import json as _json

    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
    ANTHROPIC_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")


    def _safe_select(sql: str, limit: int = 200):
        s = sql.strip().lower()
        if not s.startswith("select") or any(x in s for x in
                                             (";", " insert ", " update ", " delete ", " drop ", " alter ", " rename ",
                                              " truncate ")):
            raise ValueError("Только SELECT без ';'")
        if " limit " not in s: sql = sql.strip() + f" LIMIT {limit}"
        return ch_rows(sql)


    def _profile_cols(table: str, top_k: int = 5):
        cols = [c[0] for c in ch_rows(f"DESCRIBE {table}")]
        prof = {}
        for c in cols:
            try:
                total, nulls, uniqs = \
                ch_rows(f"SELECT count(), countIf({c} IS NULL OR toString({c})=''), uniq({c}) FROM {table}")[0]
                ex = ch_rows(f"SELECT {c} FROM {table} WHERE {c} IS NOT NULL AND toString({c})!='' LIMIT {top_k}")
                ch_type = [t[1] for t in ch_rows(f"DESCRIBE {table}") if t[0] == c][0]
                prof[c] = {"type": ch_type, "total": int(total), "nulls": int(nulls),
                           "null_rate": (nulls / total if total else 0), "uniques": int(uniqs),
                           "examples": [r[0] for r in ex]}
            except Exception as e:
                prof[c] = {"error": str(e)}
        return prof


    if not ANTHROPIC_KEY:
        st.warning("Нет CLAUDE_API_KEY/ANTHROPIC_API_KEY в окружении.")
    else:
        try:
            from anthropic import Anthropic

            anthropic_client = Anthropic(api_key=ANTHROPIC_KEY)
        except Exception as e:
            anthropic_client = None;
            st.error(f"Anthropic SDK error: {e}")

        if anthropic_client:
            if "nl_chat" not in st.session_state: st.session_state.nl_chat = []
            for role, text in st.session_state.nl_chat:
                with st.chat_message(role): st.markdown(text)

            SYSTEM = ("Ты помогаешь работать с ClickHouse. Отвечай на языке пользователя. "
                      "Если нужен расчёт — верни JSON одной строкой: "
                      '{"action":"sql","sql":"SELECT ..."} | {"action":"profile","table":"..."} | {"action":"echo","text":"..."} '
                      "Только SELECT. Добавляй LIMIT 200.")

            user_q = st.chat_input("Например: «Покажи топ-10 MCC по сумме за прошлый месяц»")
            if user_q:
                st.session_state.nl_chat.append(("user", user_q))
                with st.chat_message("user"):
                    st.markdown(user_q)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    try:
                        resp = anthropic_client.messages.create(model=CLAUDE_MODEL, max_tokens=800, temperature=0,
                                                                system=SYSTEM,
                                                                messages=[{"role": "user", "content": user_q}])
                        text = resp.content[0].text if resp and resp.content else ""
                    except Exception as e:
                        text = f'{{"action":"echo","text":"Ошибка вызова модели: {e}"}}'
                    try:
                        block = _json.loads(text.strip())
                    except Exception:
                        placeholder.markdown(text)
                    else:
                        if block.get("action") == "sql":
                            sql = block.get("sql") or ""
                            try:
                                rows = _safe_select(sql)
                                df = pd.DataFrame(rows)
                                st.code(sql, language="sql");
                                st.dataframe(df, use_container_width=True);
                                placeholder.markdown("Готово ✅")
                            except Exception as e:
                                st.error(f"Ошибка SELECT: {e}")
                        elif block.get("action") == "profile":
                            table = block.get("table") or selected_table
                            st.info(f"Профиль: **{table}**");
                            st.json(_profile_cols(table));
                            placeholder.markdown("Готово ✅")
                        elif block.get("action") == "echo":
                            placeholder.markdown(block.get("text", ""))
                        else:
                            placeholder.markdown(text)