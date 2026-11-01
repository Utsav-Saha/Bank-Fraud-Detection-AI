import streamlit as st
import pandas as pd
import numpy as np
import os

from src.graph_analysis import generate_sample_banking_network

st.set_page_config(page_title="Graph Investigation — SecureBank AI", page_icon="🕸️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0b1120; color: #f8fafc; }
.ring-card {
    background: #111827;
    border: 1px solid #dc2626;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🕸️ Graph-Based Fraud Ring & Entity Investigation")
st.markdown("Investigate suspicious account relationships, mule networks, and circular laundering rings using **Graph Analytics (BFS / DFS)**.")

st.warning("⚠️ **Dataset Architecture Notice**: The Kaggle Credit Card dataset features anonymized PCA vectors (`V1`–`V28`) without customer IDs or account numbers. This investigation module operates on a structured synthetic multi-account banking network designed specifically to demonstrate graph traversal algorithms (BFS/DFS) and laundering cycle detection.")

network = generate_sample_banking_network()
summary = network.get_graph_summary()

# Graph Metrics
g1, g2, g3, g4 = st.columns(4)
with g1:
    st.metric("Total Bank Accounts", f"{summary['total_accounts']}", "Graph Nodes")
with g2:
    st.metric("Total Inter-Account Transfers", f"{summary['total_transfers']}", "Directed Edges")
with g3:
    st.metric("Graph Density", f"{summary['density']}", "Network Connectivity")
with g4:
    st.metric("Graph Engine", summary["backend"].upper(), "Active Engine")

st.markdown("---")

tab_bfs, tab_dfs, tab_rings = st.tabs([
    "🔍 Breadth-First Search (BFS) Associate Tracing",
    "🛣️ Depth-First Search (DFS) Pathway Finder",
    "🔄 Circular Laundering Ring Detection"
])

all_accounts = sorted(list(network.graph.nodes.keys()) if hasattr(network.graph.nodes, "keys") else list(network.graph.nodes))

with tab_bfs:
    st.subheader("Breadth-First Search: Explore Connected Account Neighborhood")
    st.markdown("Identify direct counterparties (1-hop) and secondary intermediaries (2-hop associates) from a target suspect account.")
    
    col_acc, col_hop = st.columns([2, 1])
    with col_acc:
        target_account = st.selectbox("Select Target Account:", all_accounts, index=all_accounts.index("ACC-8801 (Mule A)") if "ACC-8801 (Mule A)" in all_accounts else 0)
    with col_hop:
        max_hops = st.slider("Maximum Traversal Hops (Depth):", 1, 3, 2)
        
    bfs_results = network.bfs_connected_accounts(target_account, max_hops=max_hops)
    
    if bfs_results:
        st.success(f"Discovered {len(bfs_results)} connected transfer associates within {max_hops} hops.")
        bfs_df = pd.DataFrame(bfs_results)
        bfs_df.columns = ["From Account", "To Account", "Hops Distance", "Amount ($)", "Txn ID", "Flagged Fraud"]
        st.dataframe(bfs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No outbound transfers originating from this account within the selected hop radius.")

with tab_dfs:
    st.subheader("Depth-First Search: Inter-Account Transfer Path Tracing")
    st.markdown("Uncover money flow channels between an originator account and a final beneficiary.")
    
    dfs_c1, dfs_c2 = st.columns(2)
    with dfs_c1:
        start_node = st.selectbox("Originator Account (Source):", all_accounts, index=all_accounts.index("ACC-1009 (Master)") if "ACC-1009 (Master)" in all_accounts else 0)
    with dfs_c2:
        end_node = st.selectbox("Destination Account (Target):", all_accounts, index=all_accounts.index("ACC-9999 (Collector)") if "ACC-9999 (Collector)" in all_accounts else 1)
        
    paths = network.dfs_find_paths(start_node, end_node, max_depth=5)
    
    if paths:
        st.success(f"Identified {len(paths)} distinct financial transfer pathways:")
        for idx, path in enumerate(paths, 1):
            st.markdown(f"**Path {idx}**: `{' ➔ '.join(path)}`")
    else:
        st.info(f"No direct or multi-hop path found from {start_node} to {end_node}.")

with tab_rings:
    st.subheader("Circular Laundering Ring Detector (Elementary Cycles)")
    st.markdown("Detect circular money transfers ($A \to B \to C \to A$) used to disguise funds through layering.")
    
    rings = network.detect_circular_money_rings()
    
    if rings:
        st.error(f"🚨 ALERT: {len(rings)} Circular Laundering Ring(s) Detected in Banking Network!")
        for ring in rings:
            st.markdown(f"""
            <div class="ring-card">
                <div style="font-size: 16px; font-weight: 800; color: #ef4444; margin-bottom: 6px;">
                    ⚠️ Suspicious Laundering Loop ({ring['cycle_length']} Accounts)
                </div>
                <div style="font-size: 14px; color: #f8fafc; font-family: monospace;">
                    {ring['ring_path']}
                </div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">
                    Total Cycle Volume: <b>${ring['total_volume_usd']:,.2f}</b> • Suspicious Transfers: <b>{ring['suspicious_edges']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No circular transfer cycles detected in the current network graph.")
