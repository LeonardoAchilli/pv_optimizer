import json
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==============================================================================
# SECTION 1: BACKEND LOGIC
# ==============================================================================

def get_pvgis_data(latitude: float, longitude: float) -> pd.DataFrame:
    """Fetches 15-minute interval PV generation data from PVGIS v5.2 using JSON format."""
    # Updated to PVGIS API v5.2 with JSON output
    api_url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    
    # Parameters for JSON output - much more reliable than CSV
    params = {
        'lat': latitude,
        'lon': longitude,
        'outputformat': 'json',  # Changed to JSON
        'pvcalculation': 1,
        'peakpower': 1,
        'loss': 0,
        'angle': 35,  # tilt angle
        'aspect': 0,  # azimuth (0 = south)
        'raddatabase': 'PVGIS-SARAH2',
        'startyear': 2020,
        'endyear': 2020,
        'usehorizon': 1,
        'mountingplace': 'free',
        'pvtechchoice': 'crystSi',
        'trackingtype': 0
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Extract hourly data
        if 'outputs' in data and 'hourly' in data['outputs']:
            hourly_data = data['outputs']['hourly']
            
            # Convert to DataFrame
            df = pd.DataFrame(hourly_data)
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
            df = df.set_index('time')
            
            # Convert power from W to kW
            if 'P' in df.columns:
                df['P_kW'] = df['P'] / 1000.0
            else:
                return None
            
            # Resample to 15-minute intervals if we have hourly data
            if len(df) < 35040:  # If we have hourly data (8760 hours in a year)
                df_resampled = df[['P_kW']].resample('15min').interpolate(method='linear')
                return df_resampled
            
            return df[['P_kW']]
            
        else:
            return None
            
    except requests.exceptions.HTTPError as e:
        # Try with ERA5 database as fallback
        params['raddatabase'] = 'PVGIS-ERA5'
        
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'outputs' in data and 'hourly' in data['outputs']:
                hourly_data = data['outputs']['hourly']
                df = pd.DataFrame(hourly_data)
                df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
                df = df.set_index('time')
                
                if 'P' in df.columns:
                    df['P_kW'] = df['P'] / 1000.0
                    
                    if len(df) < 35040:
                        df_resampled = df[['P_kW']].resample('15min').interpolate(method='linear')
                        return df_resampled
                    
                    return df[['P_kW']]
                else:
                    return None
            else:
                return None
                
        except Exception as e2:
            # Last resort: try without specifying database
            params.pop('raddatabase', None)
            
            try:
                response = requests.get(api_url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if 'outputs' in data and 'hourly' in data['outputs']:
                    hourly_data = data['outputs']['hourly']
                    df = pd.DataFrame(hourly_data)
                    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
                    df = df.set_index('time')
                    
                    if 'P' in df.columns:
                        df['P_kW'] = df['P'] / 1000.0
                        
                        if len(df) < 35040:
                            df_resampled = df[['P_kW']].resample('15min').interpolate(method='linear')
                            return df_resampled
                        
                        return df[['P_kW']]
                    
            except Exception as e3:
                return None
                
    except json.JSONDecodeError as e:
        return None
        
    except Exception as e:
        return None
    


def run_simulation(pv_kwp, bess_kwh_nominal, pvgis_baseline_data, consumption_profile, config):
    """Runs the full 5-year simulation with improved accuracy."""
    # Extract configuration parameters
    dod = config['bess_dod']
    c_rate = config['bess_c_rate']
    charge_eff = config['bess_charge_eff']
    discharge_eff = config['bess_discharge_eff']
    pv_degr_rate = config['pv_degradation_rate']
    bess_cal_degr_rate = config['bess_calendar_degradation_rate']
    
    # Calculate battery parameters
    usable_nominal_capacity_kwh = bess_kwh_nominal * dod
    max_charge_discharge_power_kw = bess_kwh_nominal * c_rate
    max_charge_discharge_per_step_kwh = max_charge_discharge_power_kw * 0.25  # 15-min step
    
    # Simulation parameters
    steps_per_year = len(consumption_profile)
    calendar_degr_per_step = bess_cal_degr_rate / steps_per_year
    
    # Initialize simulation variables
    soh = 1.0  # State of Health
    annual_net_savings = []
    total_grid_import = 0
    total_consumption = consumption_profile['consumption_kWh'].sum() * 5
    
    # Run 5-year simulation
    for year in range(1, 6):
        # Apply PV degradation
        pv_degradation_factor = (1 - pv_degr_rate) ** (year - 1)
        current_pv_production = pvgis_baseline_data['P_kW'] * pv_kwp * pv_degradation_factor
        
        # Initialize yearly metrics
        soc_kwh = 0.0
        yearly_energy_bought_kwh = 0
        yearly_energy_sold_kwh = 0
        
        # Simulate each 15-minute interval
        for i in range(steps_per_year):
            # Get production and consumption for this step
            prod_kwh = current_pv_production.iloc[i] * 0.25
            cons_kwh = consumption_profile['consumption_kWh'].iloc[i]
            
            # Calculate available battery capacity with current SoH
            available_capacity_kwh = usable_nominal_capacity_kwh * soh
            
            # Calculate net energy (positive = excess, negative = deficit)
            net_energy = prod_kwh - cons_kwh
            
            energy_discharged_from_bess = 0
            
            if net_energy > 0:
                # Excess energy: try to charge battery, then sell to grid
                energy_to_charge = net_energy * charge_eff
                actual_charge = min(
                    energy_to_charge,
                    available_capacity_kwh - soc_kwh,
                    max_charge_discharge_per_step_kwh
                )
                soc_kwh += actual_charge
                
                # Sell remaining excess to grid
                energy_not_charged = (net_energy * charge_eff - actual_charge) / charge_eff
                yearly_energy_sold_kwh += energy_not_charged
                
            else:
                # Energy deficit: try to discharge battery, then buy from grid
                deficit = -net_energy
                
                # Calculate how much we can discharge
                energy_from_bess_gross = min(
                    deficit / discharge_eff,
                    soc_kwh,
                    max_charge_discharge_per_step_kwh
                )
                energy_from_bess_net = energy_from_bess_gross * discharge_eff
                
                # Update battery state
                soc_kwh -= energy_from_bess_gross
                
                # Buy remaining deficit from grid
                remaining_deficit = deficit - energy_from_bess_net
                yearly_energy_bought_kwh += remaining_deficit
                
                energy_discharged_from_bess = energy_from_bess_gross
            
            # Apply battery degradation
            # Cycle degradation (based on discharge)
            if usable_nominal_capacity_kwh > 0:
                cycle_deg_this_step = (
                    (energy_discharged_from_bess / usable_nominal_capacity_kwh) * 
                    (0.2 / 7000) * 1.15  # 20% degradation after 7000 cycles, with 15% safety factor
                )
            else:
                cycle_deg_this_step = 0
            
            # Total degradation
            soh = max(0, soh - calendar_degr_per_step - cycle_deg_this_step)
        
        # Calculate annual savings
        cost_without_system = consumption_profile['consumption_kWh'].sum() * config['grid_price_buy']
        cost_with_system = yearly_energy_bought_kwh * config['grid_price_buy']
        revenue_from_exports = yearly_energy_sold_kwh * config['grid_price_sell']
        
        annual_net_savings.append(cost_without_system - cost_with_system + revenue_from_exports)
        total_grid_import += yearly_energy_bought_kwh
    
    # Project savings for years 6-10 using CAGR
    if len(annual_net_savings) > 1 and annual_net_savings[0] > 0:
        cagr = (annual_net_savings[-1] / annual_net_savings[0]) ** (1 / (len(annual_net_savings) - 1)) - 1
    else:
        cagr = 0
    
    # Project future savings
    last_real_saving = annual_net_savings[-1]
    for _ in range(5):
        next_saving = last_real_saving * (1 + cagr)
        annual_net_savings.append(next_saving)
        last_real_saving = next_saving
    
    # Calculate CAPEX
    capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290))  # Non-linear pricing
    capex_bess = bess_kwh_nominal * 150
    total_capex = capex_pv + capex_bess
    
    # Calculate O&M costs
    om_pv = (12 - 0.01 * pv_kwp) * pv_kwp  # Decreasing per-kWp cost
    om_bess = 1500 + (capex_bess * 0.015)
    total_om = om_pv + om_bess
    
    # Calculate net cash flows
    net_cash_flows = [s - total_om for s in annual_net_savings]
    
    # Calculate NPV
    wacc = config['wacc']
    npv = sum(net_cash_flows[i] / ((1 + wacc) ** (i + 1)) for i in range(10)) - total_capex
    
    # Calculate payback period
    cumulative_cash_flow = -total_capex
    payback_period = float('inf')
    for i, cash_flow in enumerate(net_cash_flows):
        cumulative_cash_flow += cash_flow
        if cumulative_cash_flow > 0:
            payback_period = i + (1 - cumulative_cash_flow / cash_flow)
            break
    
    # Calculate self-sufficiency rate
    self_sufficiency_rate = (total_consumption - total_grid_import) / total_consumption if total_consumption > 0 else 0
    
    return {
        "npv_eur": npv,
        "payback_period_years": payback_period,
        "total_capex_eur": total_capex,
        "self_sufficiency_rate": self_sufficiency_rate,
        "final_soh_percent": soh * 100,
        "annual_savings": annual_net_savings[:5],  # First 5 years actual
        "om_costs": total_om
    }


def find_optimal_system(user_inputs, config, pvgis_baseline):
    """Finds the optimal PV and BESS combination with improved search algorithm."""
    # Calculate maximum feasible sizes
    max_kwp_from_area = user_inputs['available_area_m2'] / 5.0  # 5 m²/kWp
    max_kwp_from_budget = user_inputs['budget'] / 650  # Minimum cost estimate
    max_kwp = min(max_kwp_from_area, max_kwp_from_budget)
    
    max_kwh = user_inputs['budget'] / 150  # Minimum battery cost
    
    # Define search ranges with adaptive step sizes
    kwp_step = max(5, int(max_kwp / 20))  # More granular search
    kwh_step = max(5, int(max_kwh / 20))
    
    pv_search_range = range(kwp_step, int(max_kwp) + kwp_step, kwp_step)
    bess_search_range = range(0, int(max_kwh) + kwh_step, kwh_step)
    
    # Initialize search variables
    best_result = None
    min_payback = float('inf')
    results_matrix = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    total_sims = len(pv_search_range) * len(bess_search_range)
    sim_count = 0
    
    # Search for optimal combination
    for pv_kwp in pv_search_range:
        for bess_kwh in bess_search_range:
            sim_count += 1
            progress_bar.progress(sim_count / total_sims if total_sims > 0 else 1)
            
            # Check budget constraint
            current_capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290))
            current_capex_bess = bess_kwh * 150
            
            if (current_capex_pv + current_capex_bess) > user_inputs['budget']:
                break  # Skip remaining battery sizes for this PV size
            
            # Run simulation
            result = run_simulation(pv_kwp, bess_kwh, pvgis_baseline, 
                                  user_inputs['consumption_profile_df'], config)
            
            # Store result for analysis
            result['pv_kwp'] = pv_kwp
            result['bess_kwh'] = bess_kwh
            results_matrix.append(result)
            
            # Track best result (minimum payback period)
            if result['payback_period_years'] < min_payback:
                min_payback = result['payback_period_years']
                best_result = result
                best_result['optimal_kwp'] = pv_kwp
                best_result['optimal_kwh'] = bess_kwh
    
    progress_bar.empty()
    
    # Add results matrix to best result for visualization
    if best_result:
        best_result['all_results'] = results_matrix
    
    return best_result


def build_ui():
    """Streamlit UI with enhanced features and error handling."""
    st.set_page_config(
        page_title="PV & BESS Optimizer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stMetric {
            background-color: #262626;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("⚡ Optimal PV & BESS Sizing Calculator")
    st.markdown("""
        ### Find the perfect solar + battery system for your needs
        This tool optimizes Photovoltaic (PV) and Battery Energy Storage System (BESS) sizing 
        based on your consumption data, location, and budget constraints.
    """)
    
    # Sidebar inputs
    with st.sidebar:
        st.header("🔧 Configuration")
        
        st.subheader("1. Project Constraints")
        budget = st.number_input(
            "Maximum Budget (€)",
            min_value=10000,
            max_value=500000,
            value=80000,
            step=1000,
            help="Total available budget for PV and BESS installation"
        )
        
        available_area_m2 = st.number_input(
            "Available Area for PV (m²)",
            min_value=10,
            max_value=5000,
            value=400,
            step=10,
            help="Total roof or ground area available for solar panels"
        )
        
        st.subheader("2. Location")
        st.caption("PVGIS covers Europe, Africa, and most of Asia")
        
        # Location presets
        location_preset = st.selectbox(
            "Quick location selection:",
            ["Custom", "Rome, Italy", "Berlin, Germany", "Madrid, Spain", 
             "Athens, Greece", "Cairo, Egypt", "Istanbul, Turkey"]
        )
        
        # Set coordinates based on selection
        if location_preset == "Rome, Italy":
            default_lat, default_lon = 41.9028, 12.4964
        elif location_preset == "Berlin, Germany":
            default_lat, default_lon = 52.5200, 13.4050
        elif location_preset == "Madrid, Spain":
            default_lat, default_lon = 40.4168, -3.7038
        elif location_preset == "Athens, Greece":
            default_lat, default_lon = 37.9838, 23.7275
        elif location_preset == "Cairo, Egypt":
            default_lat, default_lon = 30.0444, 31.2357
        elif location_preset == "Istanbul, Turkey":
            default_lat, default_lon = 41.0082, 28.9784
        else:
            default_lat, default_lon = 41.9028, 12.4964
        
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=default_lat, format="%.4f", 
                                min_value=-90.0, max_value=90.0,
                                help="Decimal degrees North")
        with col2:
            lon = st.number_input("Longitude", value=default_lon, format="%.4f",
                                min_value=-180.0, max_value=180.0,
                                help="Decimal degrees East")
        
        # Test location button
        if st.button("🧪 Test Location", help="Check if solar data is available for this location"):
            with st.spinner("Testing location..."):
                test_data = get_pvgis_data(lat, lon)
                if test_data is not None and not test_data.empty:
                    st.success(f"✅ Location valid! Solar data available for {lat:.2f}°N, {lon:.2f}°E")
                else:
                    st.error("❌ No solar data available for this location")
        
        st.subheader("3. Consumption Profile")
        uploaded_file = st.file_uploader(
            "Upload 15-minute consumption data (CSV)",
            type="csv",
            help="""
            The CSV must contain:
            - Column named 'consumption_kWh'
            - 35,040 rows (1 year of 15-min data)
            - Values in kWh per 15-minute interval
            """
        )
        
        # Advanced settings (collapsible)
        with st.expander("⚙️ Advanced Settings"):
            st.write("**Electricity Prices**")
            grid_buy = st.number_input("Grid Buy Price (€/kWh)", value=0.28, format="%.3f")
            grid_sell = st.number_input("Grid Sell Price (€/kWh)", value=0.05, format="%.3f")
            
            st.write("**Financial Parameters**")
            wacc = st.slider("WACC (%)", min_value=1, max_value=15, value=7) / 100
            
            st.write("**Battery Parameters**")
            dod = st.slider("Depth of Discharge (%)", 70, 95, 85) / 100
            c_rate = st.slider("C-Rate", 0.3, 1.0, 0.7, 0.1)
    expected_rows = 35041
    # Main content area
    if uploaded_file is not None:
        try:
            # Read and validate consumption data
            consumption_df = pd.read_csv(uploaded_file)
            
            if 'consumption_kWh' not in consumption_df.columns:
                st.error("❌ Error: CSV must contain a column named 'consumption_kWh'.")
                return
            
            actual_rows = len(consumption_df)+1
            
            if actual_rows != expected_rows:
                st.warning(f"""
                    ⚠️ Warning: Expected {expected_rows:,} rows but found {actual_rows:,} rows.
                    The simulation assumes 1 year of 15-minute data.
                    Results may be inaccurate.
                """)
            
            # Display consumption statistics
            st.subheader("📊 Consumption Profile Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            total_annual = consumption_df['consumption_kWh'].sum()
            with col1:
                st.metric("Annual Consumption", f"{total_annual:,.0f} kWh")
            with col2:
                st.metric("Average Daily", f"{total_annual/365:,.1f} kWh")
            with col3:
                st.metric("Peak 15-min", f"{consumption_df['consumption_kWh'].max():,.2f} kWh")
            with col4:
                st.metric("Min 15-min", f"{consumption_df['consumption_kWh'].min():,.2f} kWh")
            
            # Visualization of consumption pattern
            with st.expander("📈 View Consumption Pattern"):
                # Daily average profile
                consumption_df['hour'] = range(len(consumption_df))
                consumption_df['hour'] = (consumption_df['hour'] % 96) / 4  # Convert to hours
                hourly_avg = consumption_df.groupby('hour')['consumption_kWh'].mean() * 4  # kW
                
                st.line_chart(
                    hourly_avg,
                    use_container_width=True,
                    height=300
                )
                st.caption("Average daily consumption profile (kW)")
            
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")
            return
        
        # Run optimization button
        if st.button("🚀 Find Optimal System", type="primary", use_container_width=True):
            # Prepare inputs
            user_inputs = {
                "budget": budget,
                "available_area_m2": available_area_m2,
                "consumption_profile_df": consumption_df
            }
            
            config = {
                'bess_dod': dod if 'dod' in locals() else 0.85,
                'bess_c_rate': c_rate if 'c_rate' in locals() else 0.7,
                'bess_charge_eff': 0.95,
                'bess_discharge_eff': 0.95,
                'pv_degradation_rate': 0.01,
                'bess_calendar_degradation_rate': 0.015,
                'grid_price_buy': grid_buy if 'grid_buy' in locals() else 0.28,
                'grid_price_sell': grid_sell if 'grid_sell' in locals() else 0.05,
                'wacc': wacc if 'wacc' in locals() else 0.07
            }
            
            # Run optimization
            optimal_system = None
            with st.spinner('🔄 Fetching solar data and running optimization...'):
                # Get PVGIS data
                pvgis_baseline = get_pvgis_data(lat, lon)
                
                if pvgis_baseline is not None and not pvgis_baseline.empty:
                    st.success("✅ Solar data retrieved successfully!")
                    
                    # Run optimization
                    optimal_system = find_optimal_system(user_inputs, config, pvgis_baseline)
                else:
                    st.error("❌ Could not retrieve solar data. Please check your location or try again later.")
                    st.info("""
                    💡 **Tips:**
                    - PVGIS covers Europe, Africa, and most of Asia
                    - Americas and Oceania are not covered
                    - Try coordinates like: Rome (41.9, 12.5), Berlin (52.5, 13.4), Cairo (30.0, 31.2)
                    """)
            
            # Display results
            if optimal_system:
                st.success("✅ Optimization Complete!")
                st.markdown("---")
                
                # Key results
                st.header("🏆 Optimal System Configuration")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "PV System Size",
                        f"{optimal_system['optimal_kwp']} kWp",
                        help="Optimal photovoltaic system capacity"
                    )
                
                with col2:
                    st.metric(
                        "Battery Size",
                        f"{optimal_system['optimal_kwh']} kWh",
                        help="Optimal battery storage capacity"
                    )
                
                with col3:
                    st.metric(
                        "Payback Period",
                        f"{optimal_system['payback_period_years']:.1f} years",
                        help="Time to recover initial investment"
                    )
                
                with col4:
                    st.metric(
                        "Self-Sufficiency",
                        f"{optimal_system['self_sufficiency_rate'] * 100:.1f}%",
                        help="Percentage of consumption covered by PV+BESS"
                    )
                
                # Financial details
                st.subheader("💰 Financial Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Investment",
                        f"€{optimal_system['total_capex_eur']:,.0f}",
                        help="Total capital expenditure"
                    )
                
                with col2:
                    st.metric(
                        "10-Year NPV",
                        f"€{optimal_system['npv_eur']:,.0f}",
                        help="Net Present Value over 10 years"
                    )
                
                with col3:
                    st.metric(
                        "Annual O&M",
                        f"€{optimal_system['om_costs']:,.0f}",
                        help="Operation & Maintenance costs per year"
                    )
                
                # System health
                st.subheader("🔋 System Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Battery Health (Year 5)",
                        f"{optimal_system['final_soh_percent']:.1f}%",
                        help="Battery State of Health after 5 years"
                    )
                
                # Annual savings chart
                if 'annual_savings' in optimal_system:
                    with col2:
                        st.metric(
                            "Year 1 Savings",
                            f"€{optimal_system['annual_savings'][0]:,.0f}",
                            f"+€{optimal_system['annual_savings'][4] - optimal_system['annual_savings'][0]:,.0f} by Year 5"
                        )
                    
                    # Savings progression
                    with st.expander("📊 View Savings Progression"):
                        savings_df = pd.DataFrame({
                            'Year': range(1, 6),
                            'Net Savings (€)': optimal_system['annual_savings']
                        })
                        st.bar_chart(savings_df.set_index('Year'))
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if optimal_system['optimal_kwh'] == 0:
                    st.info("🔍 **No battery recommended** - The analysis suggests a PV-only system provides the best financial return for your situation.")
                elif optimal_system['payback_period_years'] > 8:
                    st.warning("⚠️ **Long payback period** - Consider if the environmental benefits justify the investment.")
                else:
                    st.success("✅ **Excellent investment** - The system shows strong financial returns with reasonable payback period.")
                
                # Export results
                st.markdown("---")
                results_text = f"""
                PV & BESS Optimization Results
                ==============================
                Location: {lat:.4f}°N, {lon:.4f}°E
                Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
                
                Optimal Configuration:
                - PV System: {optimal_system['optimal_kwp']} kWp
                - Battery: {optimal_system['optimal_kwh']} kWh
                - Total CAPEX: €{optimal_system['total_capex_eur']:,.0f}
                
                Financial Metrics:
                - Payback Period: {optimal_system['payback_period_years']:.1f} years
                - 10-Year NPV: €{optimal_system['npv_eur']:,.0f}
                - Annual O&M: €{optimal_system['om_costs']:,.0f}
                
                Performance:
                - Self-Sufficiency: {optimal_system['self_sufficiency_rate'] * 100:.1f}%
                - Battery SoH (Year 5): {optimal_system['final_soh_percent']:.1f}%
                """
                
                st.download_button(
                    label="📥 Download Results",
                    data=results_text,
                    file_name=f"pv_bess_optimization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
    else:
        # Instructions when no file is uploaded
        st.info("""
            📁 **Please upload a consumption data file to begin.**
            
            Your CSV file should contain:
            - A column named `consumption_kWh`
            - 35,040 rows (one full year of 15-minute interval data)
            - Consumption values in kWh per 15-minute period
            
            Example format:
            ```
            consumption_kWh
            0.125
            0.130
            0.128
            ...
            ```
            
            **Note**: 35,040 rows = 96 intervals/day × 365 days
        """)
        
        # Sample data generator
        if st.button("📊 Generate Sample Data"):
            # Create realistic consumption profile
            # 35040 = 96 intervals per day * 365 days
            hours = np.arange(0, 8760, 0.25)  # 15-min intervals for a year
            
            # Base load + daily pattern + seasonal variation + noise
            base_load = 0.3
            daily_pattern = 0.4 * np.sin((hours % 24 - 6) * np.pi / 12) ** 2
            seasonal_pattern = 0.2 * np.cos((hours / 8760) * 2 * np.pi)
            noise = np.random.normal(0, 0.05, len(hours))
            
            consumption = np.maximum(0, base_load + daily_pattern + seasonal_pattern + noise)
            
            sample_df = pd.DataFrame({
                'consumption_kWh': consumption
            })
            
            st.success(f"Generated {len(sample_df)} consumption data points")
            
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Consumption Data",
                data=csv,
                file_name="sample_consumption_data.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888;'>
            <p>Powered by PVGIS API v5.2 | Solar data © European Commission</p>
            <p>Made with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    build_ui()
