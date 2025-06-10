import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO

# ==============================================================================
# SECTION 1: BACKEND LOGIC (The functions we already built)
# ==============================================================================

def get_pvgis_data(latitude: float, longitude: float) -> pd.DataFrame:
    """Fetches 15-minute interval PV generation data from PVGIS."""
    api_url = "https://re.jrc.ec.europa.eu/api/seriescalc"
    params = {
        'lat': latitude, 'lon': longitude, 'outputformat': 'csv',
        'pvcalculation': 1, 'peakpower': 1, 'loss': 0, 'angle': 35,
        'aspect': 0, 'raddatabase': 'PVGIS-SARAH2', 'startyear': 2020,
        'endyear': 2020, 'step': 15
    }
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        lines = response.text.splitlines()
        data_start_line = 0
        for i, line in enumerate(lines):
            if line.startswith('time,P,G(i),H_sun,T2m,WS10m,Int'):
                data_start_line = i
                break
        csv_data = StringIO('\n'.join(lines[data_start_line:]))
        df = pd.read_csv(csv_data)
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
        df = df.set_index('time')
        df['P_kW'] = df['P'] / 1000.0
        return df[['P_kW']]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching PVGIS data: {e}")
        return None

def run_simulation(pv_kwp, bess_kwh_nominal, pvgis_baseline_data, consumption_profile, config):
    """Runs the full 10-year simulation. (Content is identical to previous steps)."""
    dod, c_rate, charge_eff, discharge_eff, pv_degr_rate, bess_cal_degr_rate = (
        config['bess_dod'], config['bess_c_rate'], config['bess_charge_eff'],
        config['bess_discharge_eff'], config['pv_degradation_rate'],
        config['bess_calendar_degradation_rate']
    )
    usable_nominal_capacity_kwh = bess_kwh_nominal * dod
    max_charge_discharge_power_kw = bess_kwh_nominal * c_rate
    max_charge_discharge_per_step_kwh = max_charge_discharge_power_kw * 0.25
    steps_per_year = len(consumption_profile)
    calendar_degr_per_step = bess_cal_degr_rate / steps_per_year
    soh = 1.0
    annual_net_savings = []
    total_grid_import = 0
    total_consumption = consumption_profile['consumption_kWh'].sum() * 5
    for year in range(1, 6):
        current_pv_production = pvgis_baseline_data['P_kW'] * pv_kwp * ((1 - pv_degr_rate) ** (year - 1))
        soc_kwh, yearly_energy_bought_kwh, yearly_energy_sold_kwh = 0.0, 0, 0
        for i in range(steps_per_year):
            prod_kwh = current_pv_production.iloc[i] * 0.25
            cons_kwh = consumption_profile['consumption_kWh'].iloc[i]
            available_capacity_kwh = usable_nominal_capacity_kwh * soh
            net_energy = prod_kwh - cons_kwh
            energy_discharged_from_bess = 0
            if net_energy > 0:
                energy_to_charge = net_energy * charge_eff
                actual_charge = min(energy_to_charge, available_capacity_kwh - soc_kwh, max_charge_discharge_per_step_kwh)
                soc_kwh += actual_charge
                yearly_energy_sold_kwh += (net_energy * charge_eff - actual_charge) / charge_eff
            else:
                deficit = -net_energy
                energy_from_bess_gross = min(deficit / discharge_eff, soc_kwh, max_charge_discharge_per_step_kwh)
                energy_from_bess_net = energy_from_bess_gross * discharge_eff
                soc_kwh -= energy_from_bess_gross
                yearly_energy_bought_kwh += deficit - energy_from_bess_net
                energy_discharged_from_bess = energy_from_bess_gross
            cycle_deg_this_step = ((energy_discharged_from_bess / usable_nominal_capacity_kwh) * (0.2 / 7000)) * 1.15 if usable_nominal_capacity_kwh > 0 else 0
            soh = max(0, soh - calendar_degr_per_step - cycle_deg_this_step)
        cost_without_system = consumption_profile['consumption_kWh'].sum() * config['grid_price_buy']
        cost_with_system = yearly_energy_bought_kwh * config['grid_price_buy']
        revenue_from_exports = yearly_energy_sold_kwh * config['grid_price_sell']
        annual_net_savings.append(cost_without_system - cost_with_system + revenue_from_exports)
        total_grid_import += yearly_energy_bought_kwh
    if len(annual_net_savings) > 1 and annual_net_savings[0] > 0:
        cagr = (annual_net_savings[-1] / annual_net_savings[0])**(1 / (len(annual_net_savings) - 1)) - 1
    else: cagr = 0
    last_real_saving = annual_net_savings[-1]
    for _ in range(5): next_saving = last_real_saving * (1 + cagr); annual_net_savings.append(next_saving); last_real_saving = next_saving
    capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290)); capex_bess = bess_kwh_nominal * 150
    total_capex = capex_pv + capex_bess; om_pv = (12 - 0.01 * pv_kwp) * pv_kwp
    om_bess = 1500 + (capex_bess * 0.015); total_om = om_pv + om_bess
    net_cash_flows = [s - total_om for s in annual_net_savings]
    wacc = config['wacc']; npv = sum(net_cash_flows[i] / ((1 + wacc)**(i + 1)) for i in range(10)) - total_capex
    cumulative_cash_flow = -total_capex; payback_period = float('inf')
    for i, cash_flow in enumerate(net_cash_flows):
        cumulative_cash_flow += cash_flow
        if cumulative_cash_flow > 0: payback_period = (i) + (1 - cumulative_cash_flow / cash_flow); break
    self_sufficiency_rate = (total_consumption - total_grid_import) / total_consumption if total_consumption > 0 else 0
    return {"npv_eur": npv, "payback_period_years": payback_period, "total_capex_eur": total_capex, "self_sufficiency_rate": self_sufficiency_rate, "final_soh_percent": soh * 100}

def find_optimal_system(user_inputs, config, pvgis_baseline):
    """Finds the optimal PV and BESS combination."""
    max_kwp_from_area = user_inputs['available_area_m2'] / 5.0
    max_kwp_from_budget = user_inputs['budget'] / 650
    max_kwp = min(max_kwp_from_area, max_kwp_from_budget)
    max_kwh = user_inputs['budget'] / 150
    kwp_step = max(5, int(max_kwp / 10)) # Make steps proportional to the search space
    kwh_step = max(10, int(max_kwh / 10))
    pv_search_range = range(kwp_step, int(max_kwp) + kwp_step, kwp_step)
    bess_search_range = range(0, int(max_kwh) + kwh_step, kwh_step)
    best_result, min_payback = None, float('inf')
    progress_bar = st.progress(0)
    total_sims = len(pv_search_range) * len(bess_search_range)
    sim_count = 0
    for pv_kwp in pv_search_range:
        for bess_kwh in bess_search_range:
            sim_count += 1
            progress_bar.progress(sim_count / total_sims if total_sims > 0 else 1)
            current_capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290))
            current_capex_bess = bess_kwh * 150
            if (current_capex_pv + current_capex_bess) > user_inputs['budget']: break
            result = run_simulation(pv_kwp, bess_kwh, pvgis_baseline, user_inputs['consumption_profile_df'], config)
            if result['payback_period_years'] < min_payback:
                min_payback = result['payback_period_years']
                best_result = result
                best_result['optimal_kwp'] = pv_kwp
                best_result['optimal_kwh'] = bess_kwh
    progress_bar.empty()
    return best_result

# ==============================================================================
# SECTION 2: STREAMLIT USER INTERFACE
# ==============================================================================

def build_ui():
    st.set_page_config(page_title="PV & BESS Optimizer", layout="wide")
    st.title("☀️ Optimal PV & BESS Sizing Calculator")
    st.markdown("This tool helps you find the most financially viable Photovoltaic (PV) and Battery Energy Storage System (BESS) based on your specific needs.")

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("1. Your Project Constraints")
        budget = st.number_input("Maximum Budget (€)", min_value=10000, max_value=500000, value=80000, step=1000)
        available_area_m2 = st.number_input("Available Area for PV (m²)", min_value=10, max_value=1000, value=400, step=10)
        
        st.header("2. Your Location")
        lat = st.number_input("Latitude", value=41.9, format="%.4f")
        lon = st.number_input("Longitude", value=12.5, format="%.4f")

        st.header("3. Your Consumption Profile")
        uploaded_file = st.file_uploader(
            "Upload your 15-minute consumption data (CSV)",
            type="csv",
            help="The CSV must have one column named 'consumption_kWh' with 35,040 rows for one year."
        )

    # --- Main App Logic ---
    if uploaded_file is not None:
        try:
            consumption_df = pd.read_csv(uploaded_file)
            if 'consumption_kWh' not in consumption_df.columns:
                st.error("Error: CSV must contain a column named 'consumption_kWh'.")
                return
            if len(consumption_df) != 35040:
                st.warning(f"Warning: Your file has {len(consumption_df)} rows. The simulation assumes 35,040 rows (1 year of 15-min data). Results may be inaccurate.")
        except Exception as e:
            st.error(f"Error reading or parsing CSV file: {e}")
            return
            
        if st.button("🚀 Find Optimal System"):
            # Prepare inputs for the optimizer
            user_inputs = {
                "budget": budget,
                "available_area_m2": available_area_m2,
                "consumption_profile_df": consumption_df
            }
            # System configuration (from our validated logic)
            config = {
                'bess_dod': 0.85, 'bess_c_rate': 0.7, 'bess_charge_eff': 0.95,
                'bess_discharge_eff': 0.95, 'pv_degradation_rate': 0.01,
                'bess_calendar_degradation_rate': 0.015, 'grid_price_buy': 0.28,
                'grid_price_sell': 0.05, 'wacc': 0.07
            }
            
            with st.spinner('Fetching solar data and running thousands of simulations... This may take a moment.'):
                # 1. Fetch PV data
                pvgis_baseline = get_pvgis_data(lat, lon)
                
                if pvgis_baseline is not None:
                    # 2. Run Optimizer
                    optimal_system = find_optimal_system(user_inputs, config, pvgis_baseline)
            
            st.success("Optimization Complete!")

            # --- Display Results ---
            if optimal_system:
                st.header("🏆 Optimal System Recommendation")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Optimal PV Size (kWp)", f"{optimal_system['optimal_kwp']}")
                col2.metric("Optimal BESS Size (kWh)", f"{optimal_system['optimal_kwh']}")
                col3.metric("Payback Period (Years)", f"{optimal_system['payback_period_years']:.2f}")

                st.subheader("Financial Details")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total System Cost (CAPEX)", f"€ {optimal_system['total_capex_eur']:,.2f}")
                col2.metric("Project NPV", f"€ {optimal_system['npv_eur']:,.2f}")
                
                st.subheader("Performance Metrics")
                col1, col2 = st.columns(2)
                col1.metric("Grid Self-Sufficiency", f"{optimal_system['self_sufficiency_rate'] * 100:.1f} %")
                col2.metric("BESS Final State of Health", f"{optimal_system['final_soh_percent']:.1f} %")
            else:
                st.error("No viable system could be found within your budget and area constraints. Try increasing the budget.")
    else:
        st.info("Please upload a consumption data file to begin.")

# This is the entry point of the script
if __name__ == "__main__":
    build_ui()