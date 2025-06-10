import json
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==============================================================================
# SECTION 1: BACKEND LOGIC
# ==============================================================================

def get_pvgis_data(latitude: float, longitude: float) -> pd.DataFrame:
    api_url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    params = {
        'lat': latitude, 'lon': longitude, 'outputformat': 'json',
        'pvcalculation': 1, 'peakpower': 1, 'loss': 0,
        'angle': 35, 'aspect': 0, 'raddatabase': 'PVGIS-SARAH2',
        'startyear': 2020, 'endyear': 2020, 'usehorizon': 1,
        'mountingplace': 'free', 'pvtechchoice': 'crystSi', 'trackingtype': 0
    }
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; PVOptimizerApp/1.0)'}
    db_sequence = ['PVGIS-SARAH2', 'PVGIS-ERA5', None]

    for db in db_sequence:
        current_params = params.copy()
        if db:
            current_params['raddatabase'] = db
        else:
            if 'raddatabase' in current_params: del current_params['raddatabase']

        try:
            response = requests.get(api_url, params=current_params, headers=headers, timeout=45)
            response.raise_for_status()
            data = response.json()

            if 'outputs' in data and 'hourly' in data['outputs']:
                hourly_data = data['outputs']['hourly']
                df = pd.DataFrame(hourly_data)
                if 'time' not in df.columns or 'P' not in df.columns: continue

                df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
                df = df.set_index('time')
                df['P_kW'] = df['P'] / 1000.0

                if len(df.index) == 8760: # Standard hourly data for a year
                    df_resampled = df[['P_kW']].resample('15min').interpolate(method='linear')
                    if len(df_resampled.index) == 35040: return df_resampled
                    if abs(len(df_resampled.index) - 35040) <= 4: 
                        std_index = pd.date_range(start=df_resampled.index.min().normalize(), periods=35040, freq='15min')
                        df_aligned = df_resampled.reindex(std_index, method='ffill').fillna(method='bfill')
                        if len(df_aligned) == 35040 and not df_aligned['P_kW'].isnull().any(): return df_aligned
                    continue
                elif len(df.index) == 35040: 
                    return df[['P_kW']]
                continue
            continue
        except (requests.exceptions.RequestException, json.JSONDecodeError, Exception):
            if db == db_sequence[-1]: return None
            continue
    return None

def run_simulation(pv_kwp, bess_kwh_nominal, pvgis_baseline_data, consumption_profile, config):
    dod = config['bess_dod']
    c_rate = config['bess_c_rate']
    charge_eff = config['bess_charge_eff']
    discharge_eff = config['bess_discharge_eff']
    pv_degr_rate = config['pv_degradation_rate']
    bess_cal_degr_rate = config['bess_calendar_degradation_rate']

    usable_nominal_capacity_kwh = bess_kwh_nominal * dod
    max_charge_discharge_power_kw = bess_kwh_nominal * c_rate
    max_charge_discharge_per_step_kwh = max_charge_discharge_power_kw * 0.25

    if not isinstance(consumption_profile.index, pd.RangeIndex):
        consumption_profile = consumption_profile.reset_index(drop=True)
    
    steps_per_year = len(consumption_profile)
    if steps_per_year == 0: return {"error": "Consumption profile is empty."}
    calendar_degr_per_step = bess_cal_degr_rate / steps_per_year

    current_soh = 1.0
    annual_net_savings = []
    total_grid_import_kwh_simulation = 0.0

    all_years_pv_direct_consumption_kwh = []
    all_years_pv_to_bess_kwh = []
    all_years_pv_to_grid_kwh = []
    all_years_bess_to_consumption_kwh = []
    all_years_grid_to_consumption_kwh = []
    all_years_total_pv_generation_kwh = []
    all_years_total_consumption_kwh = []
    
    pvgis_data_for_sim = pvgis_baseline_data.copy()
    if len(pvgis_data_for_sim) != steps_per_year:
        if len(pvgis_data_for_sim) == 8760 and steps_per_year == 35040:
            pvgis_data_for_sim = pvgis_data_for_sim.resample('15min').interpolate(method='linear')
            if len(pvgis_data_for_sim) != steps_per_year:
                std_index = pd.date_range(start=pvgis_data_for_sim.index.min().normalize(), periods=steps_per_year, freq='15min')
                pvgis_data_for_sim = pvgis_data_for_sim.reindex(std_index, method='ffill').fillna(method='bfill')
                if len(pvgis_data_for_sim) != steps_per_year or pvgis_data_for_sim['P_kW'].isnull().any():
                    return {"error": "PVGIS data alignment failed."}
        else:
            return {"error": f"PVGIS data length ({len(pvgis_data_for_sim)}) incompatible with consumption ({steps_per_year})."}
    pvgis_data_for_sim.index = consumption_profile.index

    total_consumption_kwh_simulation = consumption_profile['consumption_kWh'].sum() * 5.0
    consumption_kwh_annual_profile = consumption_profile['consumption_kWh'].astype(float)

    for year in range(1, 6):
        pv_degradation_factor = (1.0 - pv_degr_rate) ** (year - 1)
        current_pv_production_kw_series = pvgis_data_for_sim['P_kW'] * pv_kwp * pv_degradation_factor
        prod_kwh_series_year = current_pv_production_kw_series * 0.25
        net_energy_kwh_series_year = prod_kwh_series_year - consumption_kwh_annual_profile

        current_soc_kwh = 0.0
        energy_bought_kwh_steps = np.zeros(steps_per_year)
        energy_sold_kwh_steps = np.zeros(steps_per_year)
        actual_charge_kwh_steps = np.zeros(steps_per_year) 
        actual_discharge_net_kwh_steps = np.zeros(steps_per_year) 
        
        current_soh_value_for_step_iterations = current_soh 
        available_capacity_kwh_year = usable_nominal_capacity_kwh * current_soh_value_for_step_iterations
        if available_capacity_kwh_year < 0: available_capacity_kwh_year = 0.0
        
        for i in range(steps_per_year):
            net_energy_step = net_energy_kwh_series_year.iloc[i]
            energy_discharged_gross_for_soh_step = 0.0
            actual_charge_kwh_step_val = 0.0 
            actual_discharge_kwh_net_step_val = 0.0 

            if bess_kwh_nominal > 1e-3:
                if net_energy_step > 0:
                    energy_to_charge_eff = net_energy_step * charge_eff
                    charge_room_in_bess = max(0.0, available_capacity_kwh_year - current_soc_kwh) 
                    actual_charge_kwh_step_val = min(energy_to_charge_eff, charge_room_in_bess, max_charge_discharge_per_step_kwh)
                    current_soc_kwh += actual_charge_kwh_step_val
                    energy_sold_kwh_steps[i] = (energy_to_charge_eff - actual_charge_kwh_step_val) / charge_eff if charge_eff > 1e-6 else 0.0
                else:
                    deficit_kwh_step = -net_energy_step
                    discharge_from_bess_gross_potential = min(
                        deficit_kwh_step / discharge_eff if discharge_eff > 1e-6 else float('inf'),
                        current_soc_kwh,
                        max_charge_discharge_per_step_kwh
                    )
                    actual_discharge_kwh_net_step_val = discharge_from_bess_gross_potential * discharge_eff
                    current_soc_kwh -= discharge_from_bess_gross_potential
                    energy_discharged_gross_for_soh_step = discharge_from_bess_gross_potential
                    energy_bought_kwh_steps[i] = deficit_kwh_step - actual_discharge_kwh_net_step_val
            else:
                if net_energy_step > 0: energy_sold_kwh_steps[i] = net_energy_step
                else: energy_bought_kwh_steps[i] = -net_energy_step
            
            actual_charge_kwh_steps[i] = actual_charge_kwh_step_val
            actual_discharge_net_kwh_steps[i] = actual_discharge_kwh_net_step_val

            if bess_kwh_nominal > 1e-3 and usable_nominal_capacity_kwh > 1e-6:
                cycle_deg_this_step = 0.0
                if energy_discharged_gross_for_soh_step > 0:
                    cycle_deg_this_step = ((energy_discharged_gross_for_soh_step / usable_nominal_capacity_kwh) * (0.2 / 7000.0) * 1.15)
                current_soh_value_for_step_iterations = max(0, current_soh_value_for_step_iterations - calendar_degr_per_step - cycle_deg_this_step)
            elif bess_kwh_nominal > 1e-3 : 
                 current_soh_value_for_step_iterations = max(0, current_soh_value_for_step_iterations - calendar_degr_per_step)
        
        current_soh = current_soh_value_for_step_iterations

        current_total_pv_generation_kwh = prod_kwh_series_year.sum()
        current_total_consumption_kwh = consumption_kwh_annual_profile.sum()
        yearly_energy_sold_kwh = np.sum(energy_sold_kwh_steps)
        yearly_energy_bought_kwh = np.sum(energy_bought_kwh_steps)
        current_pv_to_grid_kwh = yearly_energy_sold_kwh
        current_grid_to_consumption_kwh = yearly_energy_bought_kwh
        current_pv_to_bess_kwh = np.sum(actual_charge_kwh_steps)
        current_bess_to_consumption_kwh = np.sum(actual_discharge_net_kwh_steps)
        pv_direct_consumption_kwh_year_step = np.minimum(prod_kwh_series_year, consumption_kwh_annual_profile)
        current_pv_direct_consumption_kwh = pv_direct_consumption_kwh_year_step.sum()

        all_years_total_pv_generation_kwh.append(current_total_pv_generation_kwh)
        all_years_total_consumption_kwh.append(current_total_consumption_kwh)
        all_years_pv_to_grid_kwh.append(current_pv_to_grid_kwh)
        all_years_grid_to_consumption_kwh.append(current_grid_to_consumption_kwh)
        all_years_pv_to_bess_kwh.append(current_pv_to_bess_kwh)
        all_years_bess_to_consumption_kwh.append(current_bess_to_consumption_kwh)
        all_years_pv_direct_consumption_kwh.append(current_pv_direct_consumption_kwh)

        grid_price_buy = float(config['grid_price_buy'])
        grid_price_sell = float(config['grid_price_sell'])
        cost_without_system_this_year = current_total_consumption_kwh * grid_price_buy
        cost_with_system_this_year = yearly_energy_bought_kwh * grid_price_buy
        revenue_from_exports_this_year = yearly_energy_sold_kwh * grid_price_sell
        annual_net_savings.append(cost_without_system_this_year - cost_with_system_this_year + revenue_from_exports_this_year)
        total_grid_import_kwh_simulation += yearly_energy_bought_kwh
    
    if not annual_net_savings: return {"error": "Simulation yielded no annual savings."}

    cagr = 0.0
    if len(annual_net_savings) > 1 and annual_net_savings[0] != 0:
        cagr_base = annual_net_savings[-1] / annual_net_savings[0]
        if cagr_base >= 0: cagr = (cagr_base) ** (1.0 / (len(annual_net_savings) -1.0)) -1.0 if len(annual_net_savings) > 1 else 0.0
    
    last_real_saving = annual_net_savings[-1] if annual_net_savings else 0.0
    for _ in range(5): 
        next_saving = last_real_saving * (1.0 + cagr)
        annual_net_savings.append(next_saving); last_real_saving = next_saving

    capex_pv = pv_kwp * (600.0 + 600.0 * np.exp(-pv_kwp / 290.0))
    capex_bess = bess_kwh_nominal * 150.0
    total_capex = capex_pv + capex_bess

    om_pv = (12.0 - 0.01 * pv_kwp) * pv_kwp
    om_bess = 1500.0 + (capex_bess * 0.015)
    total_om = om_pv + om_bess

    net_cash_flows = [s - total_om for s in annual_net_savings]
    wacc_val = float(config['wacc'])
    npv = -total_capex
    for i_flow, flow in enumerate(net_cash_flows):
        npv += flow / ((1.0 + wacc_val) ** (i_flow + 1.0))

    cumulative_cash_flow = -total_capex
    payback_period = float('inf')
    for i_pb, cash_flow_pb in enumerate(net_cash_flows):
        cumulative_cash_flow += cash_flow_pb
        if cumulative_cash_flow > 1e-6: 
            if abs(cash_flow_pb) > 1e-6 :
                 payback_period = (i_pb) + (-(cumulative_cash_flow - cash_flow_pb) / cash_flow_pb)
            else: 
                 payback_period = (i_pb + 1.0) 
            break
            
    self_sufficiency_rate = 0.0
    if total_consumption_kwh_simulation > 1e-6:
        self_sufficiency_rate = max(0.0, (total_consumption_kwh_simulation - total_grid_import_kwh_simulation) / total_consumption_kwh_simulation)

    return {
        "npv_eur": npv, "payback_period_years": payback_period, "total_capex_eur": total_capex,
        "self_sufficiency_rate": self_sufficiency_rate, "final_soh_percent": current_soh * 100.0, 
        "annual_savings": annual_net_savings[:5], "om_costs": total_om,
        "pv_direct_consumption_annual_kwh": all_years_pv_direct_consumption_kwh,
        "pv_to_bess_annual_kwh": all_years_pv_to_bess_kwh,
        "pv_to_grid_annual_kwh": all_years_pv_to_grid_kwh,
        "bess_to_consumption_annual_kwh": all_years_bess_to_consumption_kwh,
        "grid_to_consumption_annual_kwh": all_years_grid_to_consumption_kwh,
        "total_pv_generation_annual_kwh": all_years_total_pv_generation_kwh,
        "total_consumption_annual_kwh": all_years_total_consumption_kwh
    }

def find_optimal_system(user_inputs, config, pvgis_baseline):
    max_kwp_from_area = user_inputs['available_area_m2'] / 5.0
    min_pv_cost_per_kwp = 500.0
    min_bess_cost_per_kwh = 120.0
    max_kwp_from_budget = user_inputs['budget'] / min_pv_cost_per_kwp if user_inputs['budget'] > 0 else 0.0
    max_kwp = min(max_kwp_from_area, max_kwp_from_budget)
    if max_kwp < 0: max_kwp = 0.0
    
    max_kwh_from_budget = user_inputs['budget'] / min_bess_cost_per_kwh if user_inputs['budget'] > 0 else 0.0
    if max_kwh_from_budget < 0: max_kwh_from_budget = 0.0
    
    num_steps_pv = 10
    num_steps_bess = 10

    pv_min_step = max(0.5, max_kwp / num_steps_pv if max_kwp > 0.5 else 0.5)
    pv_search_values = np.arange(pv_min_step, max_kwp + pv_min_step, pv_min_step) if max_kwp > 0 else np.array([0.0])
    if max_kwp > 0 and max_kwp not in pv_search_values: pv_search_values = np.append(pv_search_values, max_kwp)
    pv_search_range = sorted(list(set(pv_search_values[pv_search_values >= 0])))

    bess_min_step = max(0.5, max_kwh_from_budget / num_steps_bess if max_kwh_from_budget > 0.5 else 0.5)
    bess_search_values = np.arange(bess_min_step, max_kwh_from_budget + bess_min_step, bess_min_step) if max_kwh_from_budget > 0 else np.array([])
    if max_kwh_from_budget > 0 and max_kwh_from_budget not in bess_search_values: bess_search_values = np.append(bess_search_values, max_kwh_from_budget)
    bess_search_range = sorted(list(set(np.concatenate(([0.0], bess_search_values[bess_search_values >= 0])))))

    best_result = None
    min_payback = float('inf')
    results_matrix = []
    
    try:
        progress_bar = st.progress(0)
    except Exception: 
        progress_bar = None

    total_sims = len(pv_search_range) * len(bess_search_range)
    sim_count = 0
    
    if total_sims == 0:
        if progress_bar: st.warning("No valid PV or BESS sizes to simulate based on constraints.")
        if progress_bar: progress_bar.empty()
        return None

    for pv_kwp_val in pv_search_range:
        if pv_kwp_val < 1e-3 and not any(s > 1e-3 for s in bess_search_range): continue

        for bess_kwh_val in bess_search_range:
            sim_count += 1
            if progress_bar: progress_bar.progress(sim_count / total_sims if total_sims > 0 else 1.0)
            
            current_capex_pv = pv_kwp_val * (600.0 + 600.0 * np.exp(-pv_kwp_val / 290.0)) if pv_kwp_val > 1e-3 else 0.0
            current_capex_bess = bess_kwh_val * 150.0 if bess_kwh_val > 1e-3 else 0.0
            
            if (current_capex_pv + current_capex_bess) > user_inputs['budget'] + 100.0: 
                if bess_kwh_val > 1e-3: break 
                elif pv_kwp_val > 1e-3 and current_capex_pv > user_inputs['budget'] + 100.0:
                    break 
            
            if pv_kwp_val < 1e-3 and bess_kwh_val < 1e-3: continue

            sim_result_data = run_simulation(pv_kwp_val, bess_kwh_val, pvgis_baseline, 
                                  user_inputs['consumption_profile_df'], config)
            
            if sim_result_data is None or "npv_eur" not in sim_result_data: continue

            sim_result_data['pv_kwp'] = pv_kwp_val
            sim_result_data['bess_kwh'] = bess_kwh_val
            results_matrix.append(sim_result_data)
            
            current_payback = sim_result_data.get('payback_period_years', float('inf'))
            current_npv = sim_result_data.get('npv_eur', -float('inf'))

            if best_result is None: 
                if current_payback != float('inf') or current_npv > -float('inf'): 
                    best_result = sim_result_data.copy()
                    min_payback = current_payback 
                    best_result['optimal_kwp'] = pv_kwp_val
                    best_result['optimal_kwh'] = bess_kwh_val
            else:
                if current_payback != float('inf') and min_payback == float('inf'):
                    is_better = True
                elif current_payback == float('inf') and min_payback != float('inf'):
                    is_better = False
                elif current_payback != float('inf') and min_payback != float('inf'):
                    if current_payback < min_payback - 0.01: 
                        is_better = True
                    elif abs(current_payback - min_payback) < 0.01: 
                        is_better = current_npv > best_result.get('npv_eur', -float('inf'))
                    else: 
                        is_better = False
                else: 
                    is_better = current_npv > best_result.get('npv_eur', -float('inf'))

                if is_better:
                    min_payback = current_payback
                    best_result = sim_result_data.copy()
                    best_result['optimal_kwp'] = pv_kwp_val
                    best_result['optimal_kwh'] = bess_kwh_val
    
    if progress_bar: progress_bar.empty()
    
    if best_result:
        best_result['all_results'] = results_matrix
    else:
        if progress_bar: st.error("No suitable system found. All options may be unviable or over budget.")

    return best_result

def build_ui():
    st.set_page_config(page_title="PV & BESS Optimizer", page_icon="☀️🔋", layout="wide", initial_sidebar_state="expanded")
    
    st.title("☀️🔋 PV & BESS System Optimizer")
    st.markdown("Determine the optimal Photovoltaic (PV) and Battery Energy Storage System (BESS) configuration based on your energy needs, location, and financial goals.")
    
    with st.sidebar:
        st.header("🛠️ System Configuration")
        st.subheader("📍 Location")
        location_options = {
            "Custom": (None, None), "Rome, Italy": (41.90, 12.50), "Berlin, Germany": (52.52, 13.41),
            "Madrid, Spain": (40.42, -3.70), "Paris, France": (48.86, 2.35), 
            "London, UK": (51.51, -0.13), "New York, USA": (40.71, -74.01),
            "Tokyo, Japan": (35.68, 139.70),"Sydney, Australia": (-33.87, 151.21)
        }
        loc_preset = st.selectbox("Quick Location:", list(location_options.keys()), index=0, key="loc_preset_sb")
        def_lat, def_lon = location_options[loc_preset]
        if def_lat is None: 
            def_lat = st.session_state.get('ui_lat', 41.90); def_lon = st.session_state.get('ui_lon', 12.50)
        ui_lat = st.number_input("Latitude (°N)", value=def_lat, format="%.2f", min_value=-90.0, max_value=90.0, key="ui_lat_ni")
        ui_lon = st.number_input("Longitude (°E)", value=def_lon, format="%.2f", min_value=-180.0, max_value=180.0, key="ui_lon_ni")
        st.session_state.ui_lat = ui_lat; st.session_state.ui_lon = ui_lon
        if st.button("🌍 Test PVGIS Data", help="Verify data availability for the entered coordinates.", key="test_pvgis_btn"):
            with st.spinner(f"Querying PVGIS for ({ui_lat:.2f}, {ui_lon:.2f})..."):
                pvg_data = get_pvgis_data(ui_lat, ui_lon)
                if pvg_data is not None and not pvg_data.empty: st.success(f"✅ PVGIS data found ({len(pvg_data)} points).")
                else: st.error("❌ PVGIS data not found. Check coordinates or PVGIS service status.")
        st.subheader("💰 Financial Constraints")
        ui_budget = st.number_input("Max Budget (€)", value=50000, min_value=1000, step=500, key="ui_budget_ni")
        ui_area = st.number_input("Max PV Area (m²)", value=100, min_value=5, step=5, key="ui_area_ni")
        st.subheader("📄 Consumption Data")
        ui_uploaded_file = st.file_uploader("Upload 15-min Consumption CSV", type="csv", help="CSV: 'consumption_kWh' column, 35040 rows for 1 year.", key="ui_uploader")
        with st.expander("⚙️ Advanced Economic & Technical Parameters"):
            st.markdown("**Grid Interaction (€/kWh)**"); c1, c2 = st.columns(2)
            ui_grid_buy = c1.number_input("Purchase Price", value=0.30, format="%.3f", step=0.01, key="ui_gb_ni")
            ui_grid_sell = c2.number_input("Feed-in Tariff", value=0.06, format="%.3f", step=0.01, key="ui_gs_ni")
            st.markdown("**Discounting & Degradation (%)**"); c1, c2 = st.columns(2)
            ui_wacc_pct = c1.slider("WACC", 1.0, 15.0, 7.0, 0.5, format="%.1f%%", key="ui_wacc_sl")
            ui_pv_degr_pct = c2.slider("PV Degradation/yr", 0.1, 2.0, 0.5, 0.1, format="%.1f%%", key="ui_pvd_sl")
            st.markdown("**Battery System Details**"); c1,c2 = st.columns(2)
            ui_bess_dod_pct = c1.slider("Usable DoD", 70, 100, 90, format="%d%%", key="ui_dod_sl")
            ui_bess_c_rate = c2.slider("C-Rate (Charge/Discharge)", 0.2, 1.0, 0.5, 0.05, key="ui_crate_sl")
            c1,c2 = st.columns(2)
            ui_bess_charge_eff_pct = c1.slider("Charge Efficiency", 80, 99, 95, format="%d%%", key="ui_bce_sl")
            ui_bess_discharge_eff_pct = c2.slider("Discharge Efficiency", 80, 99, 95, format="%d%%", key="ui_bde_sl")
            ui_bess_cal_degr_pct = st.slider("Calendar Degradation/yr", 0.5, 5.0, 2.0, 0.1, format="%.1f%%", key="ui_bcd_sl")
    
    if ui_uploaded_file:
        try:
            df_consumption = pd.read_csv(ui_uploaded_file)
            if 'consumption_kWh' not in df_consumption.columns: st.error("❌ CSV must have 'consumption_kWh' column."); return
            if not pd.api.types.is_numeric_dtype(df_consumption['consumption_kWh']): st.error("❌ 'consumption_kWh' data must be numeric."); return
            if df_consumption['consumption_kWh'].isnull().any():
                st.warning("⚠️ Consumption data has NaNs, filled with 0."); df_consumption['consumption_kWh'] = df_consumption['consumption_kWh'].fillna(0.0)
            st.success(f"Successfully loaded {len(df_consumption)} consumption data rows.")
            if len(df_consumption) != 35040: st.warning(f"⚠️ Expected 35040 rows for a full year, got {len(df_consumption)}. Results may be indicative.")

            if st.button("🚀 Optimize System Configuration", type="primary", use_container_width=True, key="optimize_btn"):
                inputs_dict = {"budget": float(ui_budget), "available_area_m2": float(ui_area), "consumption_profile_df": df_consumption.copy()}
                config_sim = {
                    'bess_dod': float(ui_bess_dod_pct)/100.0, 'bess_c_rate': float(ui_bess_c_rate),
                    'bess_charge_eff': float(ui_bess_charge_eff_pct)/100.0, 'bess_discharge_eff': float(ui_bess_discharge_eff_pct)/100.0,
                    'pv_degradation_rate': float(ui_pv_degr_pct)/100.0, 'bess_calendar_degradation_rate': float(ui_bess_cal_degr_pct)/100.0,
                    'grid_price_buy': float(ui_grid_buy), 'grid_price_sell': float(ui_grid_sell), 'wacc': float(ui_wacc_pct)/100.0
                }
                with st.spinner("Crunching numbers... This might take a moment!"):
                    pvgis_df = get_pvgis_data(ui_lat, ui_lon)
                    if pvgis_df is None or pvgis_df.empty: st.error("Failed to get PVGIS data. Cannot optimize."); return
                    st.info(f"PVGIS data for ({ui_lat:.2f}, {ui_lon:.2f}) loaded. Starting optimization...")
                    final_results = find_optimal_system(inputs_dict, config_sim, pvgis_df)

                if final_results and "npv_eur" in final_results:
                    st.balloons()
                    st.header("📈 Optimal System Results")
                    r_opt_pv = final_results.get('optimal_kwp',0.0); r_opt_bess = final_results.get('optimal_kwh',0.0)
                    r_payback = final_results.get('payback_period_years', float('inf')); r_npv = final_results.get('npv_eur',0.0)
                    r_capex = final_results.get('total_capex_eur',0.0); r_self_suff = final_results.get('self_sufficiency_rate',0.0)*100.0
                    r_soh = final_results.get('final_soh_percent',0.0)
                    r_savings1 = final_results.get('annual_savings',[0.0])[0] if final_results.get('annual_savings') else 0.0
                    st.markdown(f"**Recommended Configuration: {r_opt_pv:.1f} kWp PV | {r_opt_bess:.1f} kWh BESS**")
                    cols_metrics1 = st.columns(3)
                    cols_metrics1[0].metric("Payback Period", f"{r_payback:.1f} years" if r_payback != float('inf') else ">10 years")
                    cols_metrics1[1].metric("10-Year NPV", f"€{r_npv:,.0f}")
                    cols_metrics1[2].metric("Initial CAPEX", f"€{r_capex:,.0f}")
                    cols_metrics2 = st.columns(3)
                    cols_metrics2[0].metric("Self-Sufficiency", f"{r_self_suff:.1f}%")
                    cols_metrics2[1].metric("BESS SoH (Yr 5)", f"{r_soh:.1f}%")
                    cols_metrics2[2].metric("Year 1 Savings", f"€{r_savings1:,.0f}")

                    st.subheader("⚡ Annual Energy Flow Analysis (Optimal System)")
                    year_options = [f"Year {i+1}" for i in range(5)]
                    selected_year_str = st.selectbox("Select Year for Detailed Flow:", year_options, key="year_flow_select")
                    selected_year_idx = year_options.index(selected_year_str)

                    pv_direct_annual = final_results.get("pv_direct_consumption_annual_kwh", [0.0]*5)
                    pv_to_bess_annual = final_results.get("pv_to_bess_annual_kwh", [0.0]*5)
                    pv_to_grid_annual = final_results.get("pv_to_grid_annual_kwh", [0.0]*5)
                    bess_to_cons_annual = final_results.get("bess_to_consumption_annual_kwh", [0.0]*5)
                    grid_to_cons_annual = final_results.get("grid_to_consumption_annual_kwh", [0.0]*5)
                    total_pv_gen_annual = final_results.get("total_pv_generation_annual_kwh", [0.0]*5)
                    total_cons_annual = final_results.get("total_consumption_annual_kwh", [0.0]*5)

                    pv_direct_selected = pv_direct_annual[selected_year_idx] if selected_year_idx < len(pv_direct_annual) else 0.0
                    pv_to_bess_selected = pv_to_bess_annual[selected_year_idx] if selected_year_idx < len(pv_to_bess_annual) else 0.0
                    pv_to_grid_selected = pv_to_grid_annual[selected_year_idx] if selected_year_idx < len(pv_to_grid_annual) else 0.0
                    bess_to_cons_selected = bess_to_cons_annual[selected_year_idx] if selected_year_idx < len(bess_to_cons_annual) else 0.0
                    grid_to_cons_selected = grid_to_cons_annual[selected_year_idx] if selected_year_idx < len(grid_to_cons_annual) else 0.0
                    total_pv_gen_selected_year = total_pv_gen_annual[selected_year_idx] if selected_year_idx < len(total_pv_gen_annual) else 0.0
                    total_consumption_selected_year = total_cons_annual[selected_year_idx] if selected_year_idx < len(total_cons_annual) else 0.0
                    
                    col_flow_metrics1, col_flow_metrics2 = st.columns(2)
                    col_flow_metrics1.metric(f"Total PV Generation ({selected_year_str})", f"{total_pv_gen_selected_year:,.0f} kWh")
                    col_flow_metrics2.metric(f"Total Consumption ({selected_year_str})", f"{total_consumption_selected_year:,.0f} kWh")
                    
                    st.markdown(f"**Energy Consumption Breakdown ({selected_year_str})**")
                    sources_data = {'Source': ["PV Directly Used", "From Battery", "From Grid"], 
                                    'Energy (kWh)': [pv_direct_selected, bess_to_cons_selected, grid_to_cons_selected]}
                    df_sources = pd.DataFrame(sources_data)
                    st.bar_chart(df_sources.set_index('Source'), height=300)

                    st.markdown(f"**PV Energy Allocation ({selected_year_str})**")
                    pv_alloc_data = {'Allocation': ["Directly Used by Load", "Charged to Battery", "Exported to Grid"],
                                     'Energy (kWh)': [pv_direct_selected, pv_to_bess_selected, pv_to_grid_selected]}
                    df_pv_alloc = pd.DataFrame(pv_alloc_data)
                    st.bar_chart(df_pv_alloc.set_index('Allocation'), height=300)
                    
                    with st.expander("📊 View NPV & Payback Sensitivity Plots", expanded=False):
                        try:
                            all_results_list = final_results.get('all_results')
                            if all_results_list and isinstance(all_results_list, list) and len(all_results_list) > 0:
                                df_results = pd.DataFrame(all_results_list)
                                
                                # NPV Plot
                                if all(col in df_results.columns for col in ['pv_kwp', 'bess_kwh', 'npv_eur']):
                                    st.subheader("NPV vs. PV System Size (for different BESS sizes)")
                                    df_pivot_npv = df_results.pivot(index='pv_kwp', columns='bess_kwh', values='npv_eur')
                                    df_pivot_npv.columns = [f"{col:.0f} kWh BESS" for col in df_pivot_npv.columns]
                                    st.line_chart(df_pivot_npv, height=400)
                                else:
                                    st.info("NPV sensitivity data is missing required columns (pv_kwp, bess_kwh, npv_eur).")

                                # Payback Period Plot
                                if all(col in df_results.columns for col in ['pv_kwp', 'bess_kwh', 'payback_period_years']):
                                    st.subheader("Payback Period vs. PV System Size (for different BESS sizes)")
                                    df_pivot_payback = df_results.pivot(index='pv_kwp', columns='bess_kwh', values='payback_period_years')
                                    df_pivot_payback.columns = [f"{col:.0f} kWh BESS" for col in df_pivot_payback.columns]
                                    st.line_chart(df_pivot_payback, height=400)
                                    st.caption("Note: Infinite payback periods may not be distinctly visualized or may be clipped by the chart's automatic y-axis scaling.")
                                else:
                                    st.info("Payback period sensitivity data is missing required columns (pv_kwp, bess_kwh, payback_period_years).")
                            else:
                                st.info("No sensitivity data available to plot (simulation might have run for a single point or data is missing).")
                        except Exception as e_plot_sensitivity:
                            st.error(f"Error plotting sensitivity graphs: {e_plot_sensitivity}")


                    with st.expander("View Simulation Parameters & Full Results Matrix (Raw)"): 
                        st.subheader("Simulation Parameters Used (Optimal Run):") 
                        st.json({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in config_sim.items()})
                        if 'all_results' in final_results and final_results['all_results']:
                            st.subheader("All Simulated Scenarios (Raw Data):") 
                            df_all_res_raw = pd.DataFrame(final_results['all_results']) 
                            st.dataframe(df_all_res_raw) 
                            try:
                                csv_export = df_all_res_raw.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Download All Scenarios (CSV)", csv_export, "all_simulation_results.csv", "text/csv", key="dl_all_sim_btn")
                            except Exception as e_csv: st.error(f"Error creating CSV: {e_csv}")
                        else: st.info("No detailed scenario matrix (raw data) available.")

                elif final_results and "error" in final_results:
                     st.error(f"Simulation Error: {final_results['error']}")
                else:
                    st.error("Optimization did not yield a conclusive result. Please check inputs or try different parameters.")
        
        except Exception as e_main:
            st.error(f"An unexpected error occurred: {str(e_main)}")

    else:
        st.info("Upload your 15-minute interval consumption data (CSV) to begin the optimization.")
        if st.button("📋 Generate Sample Consumption CSV", key="gen_sample_btn"):
            sample_cons = np.random.rand(35040) * 0.5 + 0.1 
            df_sample = pd.DataFrame({'consumption_kWh': sample_cons})
            csv_sample = df_sample.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Sample CSV", csv_sample, "sample_consumption.csv", "text/csv", key="dl_sample_btn")
            st.success("Sample CSV generated and ready for download.")

    st.markdown("--- 
*Disclaimer: This tool provides estimates for informational purposes only. Consult professionals for financial decisions.*")

if __name__ == "__main__":
    build_ui()
