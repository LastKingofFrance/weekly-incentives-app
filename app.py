# app.py
import streamlit as st
from process import process_files  # this uses the updated process.py

st.set_page_config(page_title="Weekly Incentives App", layout="wide")

st.title("📊 Weekly Incentives Calculation")

st.markdown("Upload the 4 required Excel files for this week's incentive calculation:")

# Upload section
uploaded_nbt = st.file_uploader("Upload NBT File", type=["xlsx"], key="nbt")
uploaded_agency_unit = st.file_uploader("Upload 'Agency and Unit' File", type=["xlsx"], key="agency_unit")
uploaded_active_agents = st.file_uploader("Upload 'Active Agents' File", type=["xlsx"], key="active_agents")
uploaded_master = st.file_uploader("Upload 'Master' File", type=["xlsx"], key="master")

# Once all are uploaded
if uploaded_nbt and uploaded_agency_unit and uploaded_active_agents and uploaded_master:
    if st.button("⚙️ Process Incentives"):
        with st.spinner("Processing..."):
            try:
                output_excel, report_filename = process_files(
                    nbt_file=uploaded_nbt,
                    agency_unit_file=uploaded_agency_unit,
                    active_agents_file=uploaded_active_agents,
                    master_file=uploaded_master
                )
                st.success("✅ Done! Download your report below:")
                st.download_button("📥 Download Incentive Report", data=output_excel, file_name=report_filename)
            except Exception as e:
                st.error(f"❌ An error occurred during processing:\n\n{str(e)}")
else:
    st.warning("⬆️ Please upload all 4 files to proceed.")
