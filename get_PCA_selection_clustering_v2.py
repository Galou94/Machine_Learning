#!/usr/bin/env python
import os
import glob
import lattice
import ase
import hdbscan
import numpy               as     np
import Chebyshev           as     chb  # Custom module containing the FingerprintBasis class
import matplotlib.pyplot   as     plt
from sklearn.decomposition import PCA
from sklearn.cluster       import KMeans, MiniBatchKMeans, dbscan
from sklearn.preprocessing import StandardScaler
from kneed                 import KneeLocator
from ase.io                import read, write

# =============================================================================
# Descriptor and PCA Functions
# =============================================================================

def get_Chebyshev_descriptor(atoms):
    """
    Computes the Chebyshev-based descriptor for each atom in an ASE Atoms object,
    using the Lattice object to determine neighbors under periodic boundary conditions.
    
    Parameters:
      atoms (ase.Atoms): One atomic configuration.
    
    Returns:
      descriptor_matrix (numpy.ndarray): Array of shape (n_atoms, D) where D is the descriptor length.
    """
    # --- Descriptor parameters and mapping ---
    mapping       = {'N': 0, 'O': 1, 'C': 2}
    num_types     = 3
    atom_types    = ['N', 'O', 'C']
    radial_order  = 16
    angular_order = 12
    radial_Rc     = 6.0
    angular_Rc    = 6.0

    # Initialize the fingerprint basis (Chebyshev descriptor)
    sfb = chb.FingerprintBasis(num_types, atom_types, radial_order,
                               angular_order, radial_Rc, angular_Rc)
    # sfb.print_info()  # Uncomment to display descriptor details

    # --- Set up the Lattice object for neighbor searching ---
    cell = atoms.get_cell()
    lat_obj = lattice.Lattice(cell)
    max_cutoff = max(radial_Rc, angular_Rc)

    n_atoms = len(atoms)
    positions = atoms.get_positions()  # Cartesian coordinates

    # Convert Cartesian positions to fractional coordinates
    frac_positions = lat_obj.get_fractional_coords(positions)

    descriptor_matrix = np.zeros((n_atoms, sfb.N))
    symbols = atoms.get_chemical_symbols()

    for i in range(n_atoms):
        coo0 = positions[i]
        neighbors = lat_obj.get_points_in_sphere(frac_positions, coo0, max_cutoff, zip_results=True)
        neighbor_types = []
        neighbor_positions = []
        for shifted_frac, dist, idx in neighbors:
            if np.isclose(dist, 0.0):
                continue
            neighbor_types.append(mapping[symbols[idx]])
            neighbor_cart = lat_obj.get_cartesian_coords(shifted_frac)
            neighbor_positions.append(neighbor_cart)
        if len(neighbor_positions) == 0:
            d = np.zeros(sfb.N)
        else:
            neighbor_positions = np.array(neighbor_positions)
            d = sfb.eval(coo0, neighbor_types, neighbor_positions)
        descriptor_matrix[i, :] = d

    return descriptor_matrix

def get_PCA_varianceAnalyze(descriptors, variance_threshold=0.95, plot_filename=None):
    """
    Performs PCA on the Chebyshev descriptors to determine the number of components
    needed to capture a specified fraction of the total variance. Also saves the plot.
    
    Parameters:
      descriptors (numpy.ndarray): 2D array of feature vectors (NxD).
      variance_threshold (float): Fraction of variance to retain (default 0.95).
      plot_filename (str): Filename to save the PCA variance plot (if provided).
    
    Returns:
      optimal_components (int): Number of principal components required.
    """
    pca = PCA()
    pca.fit(descriptors)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_components  = np.argmax(cumulative_variance >= variance_threshold) + 1

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="--")
    plt.axhline(y=variance_threshold, color="r", linestyle="--", label=f"{variance_threshold*100:.0f}% Variance")
    plt.axvline(x=optimal_components, color="g", linestyle="--", label=f"Optimal Components: {optimal_components}")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance Analysis")
    plt.legend()
    if plot_filename is not None:
        plt.savefig(plot_filename)
        plt.close()
        print(f"PCA variance analysis plot saved to {plot_filename}")
    else:
        plt.show()

    return optimal_components

def get_optimal_clusters(X, k_min=15000, k_max=20000, step=1000, plot=True, plot_filename=None):
    """
    Determines the optimal number of clusters using the Elbow Method with MiniBatchKMeans.
    
    Parameters:
      X : array-like, shape (n_samples, n_features)
          Data to cluster.
      k_min : int, optional (default=10)
          Minimum number of clusters to evaluate.
      k_max : int, optional (default=100)
          Maximum number of clusters to evaluate.
      step : int, optional (default=1000)
          Step between k values.
      plot : bool, optional (default=False)
          If True, displays a plot.
      plot_filename : str, optional (default=None)
          If provided, the plot is saved to this filename.
    
    Returns:
      optimal_k : int or None
          Estimated optimal number of clusters (elbow point).
      inertias : list
          Inertia values for each k.
    """
    inertias = []
    k_values = list(range(k_min, k_max + 1, step))
    
    # Compute inertia for each number of clusters using MiniBatchKMeans.
    for k in k_values:
        mbk = MiniBatchKMeans(n_clusters=k, random_state=42)
        mbk.fit(X)
        inertias.append(mbk.inertia_)
    
    # Determine the elbow point using KneeLocator.
    kl = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    # Plot the inertia vs. k curve.
    if plot or plot_filename is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, inertias, marker="o", linestyle="--")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (WCSS)")
        plt.title("Elbow Method for Optimal k")
        if optimal_k is not None:
            plt.axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal k = {optimal_k}")
            plt.legend()
        if plot_filename is not None:
            plt.savefig(plot_filename)
            plt.close()
            print(f"Elbow plot saved to {plot_filename}")
        else:
            plt.show()
    
    return optimal_k, inertias

def get_kmeans_selection(pca_components, n_clusters):
    """
    Performs k-means clustering on the given data to select representative configurations.
    
    Parameters:
      pca_components (numpy.ndarray): PCA-transformed data.
      n_clusters (int): Desired number of clusters.
    
    Returns:
      selected_indices (list): Indices of selected configurations.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_components)

    selected_indices = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        centroid = kmeans.cluster_centers_[cluster]
        distances = np.linalg.norm(pca_components[cluster_indices] - centroid, axis=1)
        selected_index = cluster_indices[np.argmin(distances)]
        selected_indices.append(selected_index)

    return selected_indices

def get_hdbscan_selection(pca_components, min_cluster_size=10, min_samples=None):
    """
    Performs density-based clustering on the entire PCA-transformed data using HDBSCAN,
    and selects representative configurations.
    
    Parameters:
      pca_components (numpy.ndarray): PCA-transformed data (n_samples, n_features).
      min_cluster_size (int): The minimum size of clusters; HDBSCAN parameter.
      min_samples (int, optional): The number of samples in a neighborhood for a point 
                                   to be considered as a core point. If None, defaults 
                                   to the value of min_cluster_size.
    
    Returns:
      selected_indices (list): List of indices corresponding to selected configurations.
    """
    # Initialize HDBSCAN clusterer.
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(pca_components)
    
    selected_indices = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:
            # Include all noise points. Alternatively, you might sample only a subset.
            noise_indices = np.where(labels == label)[0]
            selected_indices.extend(noise_indices)
        else:
            # For each cluster, find the configuration closest to the cluster centroid.
            cluster_indices = np.where(labels == label)[0]
            cluster_points = pca_components[cluster_indices]
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            representative_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(representative_idx)
    
    # Optionally, sort indices.
    selected_indices = sorted(selected_indices)
    return selected_indices


def get_2D_PCA_components_selection_graph(pca_components, selected_indices, plot_filename):
    """
    Creates and saves a 2D PCA projection plot of the configurations, highlighting the selected ones.
    
    Parameters:
      pca_components (numpy.ndarray): PCA-transformed data (n_samples, n_components).
      selected_indices (list or array-like): Indices of configurations selected by clustering.
      plot_filename (str): Full filename (including path) where the plot will be saved.
    
    Returns:
      pca_components_2d (numpy.ndarray): The 2D PCA projection of the configurations.
    """
    # Extract the first two principal components.
    pca_components_2d = pca_components[:, :2]
    
    # Create the scatter plot.
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_components_2d[:, 0], pca_components_2d[:, 1], alpha=0.2, label="All Configurations")
    plt.scatter(pca_components_2d[selected_indices, 0], pca_components_2d[selected_indices, 1],
                color="red", label="Selected Configurations")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA 2D Projection with Selected Configurations")
    plt.legend()
    
    # Save and close the plot.
    plt.savefig(plot_filename)
    plt.close()
    print(f"PCA selection plot saved to {plot_filename}")
    
    return pca_components_2d

def get_2D_PCA_components_selection_graph_with_Cmap(selected_pca_components, energies, plot_filename):
    """
    Generates and saves a 2D scatter plot of the selected PCA components,
    where each point is colored according to its potential energy.
    
    Parameters:
      selected_pca_components (numpy.ndarray): PCA-transformed data (n_samples, n_components)
                                                 for the selected configurations. The first two
                                                 components will be used for the 2D projection.
      energies (array-like): Potential energy values corresponding to each selected configuration.
      plot_filename (str): Full filename (including path) where the plot will be saved.
    
    Returns:
      pca_components_2d (numpy.ndarray): The 2D PCA projection (first two components) of the selected configurations.
    """
    # Extract the first two principal components for the 2D projection.
    pca_components_2d = selected_pca_components[:, :2]
    
    # Create the scatter plot.
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_components_2d[:, 0], pca_components_2d[:, 1],
                          c=energies, cmap="viridis", alpha=0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Potential Energy")
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D PCA Projection with Potential Energy")
    
    # Save and close the plot.
    plt.savefig(plot_filename)
    plt.close()
    print(f"2D PCA graph with potential energy colormap saved to {plot_filename}")

# =============================================================================
# Core Processing Functions
# =============================================================================

def process_structures(structures, descriptors_matrix, base, output_dir, variance_threshold=0.95, k_min=15000, k_max=20000, clustering='K_means'):
    """
    Processes a list of ASE Atoms objects:
      - Computes Chebyshev descriptors.
      - Runs PCA variance analysis to determine the required number of components.
      - Determines the optimal number of clusters via the Elbow Method.
      - Performs clustering-based selection.
      - Creates a 2D PCA projection plot with selected configurations marked.
      - Saves the plot and the selected configurations to a .traj file.
    
    Parameters:
      structures (list): List of ASE Atoms objects.
      descriptors_matrix (numpy.ndarray): Array of Chebyshev descriptors (one per structure).
      base (str): Base name for saving files.
      output_dir (str): Directory where output files will be saved.
      variance_threshold (float): Variance threshold for PCA analysis.
      k_min (int): Minimum number of clusters to evaluate.
      k_max (int): Maximum number of clusters to evaluate.
    """
    # Convert descriptor matrix into a vector for each configuration.
    descriptors                = np.array([d.flatten() for d in descriptors_matrix])
    
    # Standardize the descriptor data: each feature is scaled to have zero mean and unit variance,
    # which improves the performance and convergence of many machine learning algorithms. 
    scaler                     = StandardScaler()
    descriptors_scaled         = scaler.fit_transform(descriptors)

    # --- PCA Variance Analysis ---
    # This is done to reduce the number of components describing the configurations and make faster(easier) the clsutering selection
    pca_variance_plot_file     = os.path.join(output_dir, base + "_PCA_varianceAnalysis.pdf")
    optimal_components         = get_PCA_varianceAnalyze(descriptors_scaled, variance_threshold=variance_threshold, plot_filename=pca_variance_plot_file)
    print(f"Optimal PCA components for {base}: {optimal_components}")

    # Compute PCA projection using the optimal number of components.
    pca                        = PCA(n_components=optimal_components)
    pca_components             = pca.fit_transform(descriptors_scaled)

    if clustering == 'K_means':
        # --- K Means-Based Selection ---
        # Determine Optimal Clusters Using the Elbow Method
        elbow_plot_file        = os.path.join(output_dir, base + "_Elbow_Kmeans.pdf")
       # optimal_clusters, _    = get_optimal_clusters(pca_components, k_min=k_min, k_max=k_max, plot=True, plot_filename=elbow_plot_file)
        optimal_clusters = 15000
        # Clusters selection
        selected_indices       = get_kmeans_selection(pca_components, n_clusters=optimal_clusters)
    elif clustering == 'hdbscan':
        # --- hdbscan-Based Selection ---
        selected_indices       = get_hdbscan_selection(pca_components, min_cluster_size=18000, min_samples=None)
        
    # --- 2D PCA Projection for Visualization ---
    selection_plot_file        = os.path.join(output_dir, base + f"{len(selected_indices)}cfg_2D_PCA_cluster_selection_graph_{clustering}.pdf")
    get_2D_PCA_components_selection_graph(pca_components, selected_indices, selection_plot_file)

    # --- Save Selected Configurations ---
    selected_structures        = [structures[i] for i in selected_indices]
    selected_traj_file         = os.path.join(output_dir, f"{base}_{len(selected_structures)}_selected_structures_{clustering}.traj")
    write(selected_traj_file, selected_structures)
    print(f"Saved {len(selected_structures)} selected configurations to {selected_traj_file}")

    # --- Save Selected Descriptors as .npy File ---
    # Extract the descriptors corresponding to the selected configurations.
    selected_descriptors       = [descriptors[i] for i in selected_indices]
    selected_descriptors_file  = os.path.join(output_dir, f"{base}_{len(selected_structures)}_selected_descriptors_{clustering}.npy")
    np.save(selected_descriptors_file, selected_descriptors)
    print(f"Saved selected descriptors to {selected_descriptors_file}")

    # --- Save PCA Components as .npy File ---    
    selected_pca_components    = [pca_components[i] for i in selected_indices]
    pca_components_file        = os.path.join(output_dir, f"{base}_{len(selected_structures)}_selected_pca_components_{clustering}.npy")
    np.save(pca_components_file, selected_pca_components)
    print(f"Saved selected PCA components to {pca_components_file}")

    # --- 2D PCA Projection for Visualization ---
    plot_file_with_energy      = os.path.join(output_dir, base + f"{len(selected_indices)}cfg_2D_PCA_graph_with_Epot_{clustering}.pdf")
    # Compute the potential energies for the selected configurations
    energies = np.array([atoms.get_potential_energy() for atoms in selected_structures])
    selected_pca_components = np.array(selected_pca_components)
    get_2D_PCA_components_selection_graph_with_Cmap(selected_pca_components, energies, plot_file_with_energy)

def get_structures_and_descriptors(traj_file, descriptor_file="CHB_descriptors.npy"):
    """
    Computes (or loads cached) Chebyshev descriptors for each structure in a .traj file.
    
    Parameters:
      traj_file (str): Path to the .traj file.
      descriptor_file (str): Filename for saving/loading descriptors.
    
    Returns:
      structures (list): List of ASE Atoms objects.
      descriptors (numpy.ndarray): Array of descriptors (one per structure).
    """
    
    print(f"\nLoading trajectory from {traj_file}...")
    structures = read(traj_file, index=":")
    if os.path.exists(descriptor_file):
        print(f"Loading cached descriptors from {descriptor_file}...")
        descriptors = np.load(descriptor_file, allow_pickle=True)
    else:
        print(f"Computing Chebyshev descriptors for {traj_file}")
        descriptors = np.array([get_Chebyshev_descriptor(atoms) for atoms in structures], dtype=object)
        np.save(descriptor_file, descriptors)
        print(f"Saved computed descriptors to {descriptor_file}.")
    
    return structures, descriptors

def process_traj_file(traj_file, variance_threshold=0.95, output_dir="."):
    """
    Processes a single trajectory file.
    
    Parameters:
      traj_file (str): Path to the .traj file.
      variance_threshold (float): Variance threshold for PCA analysis.
      output_dir (str): Directory where output files will be saved.
    """
    print(f"Processing file: {traj_file}")
    base = os.path.splitext(os.path.basename(traj_file))[0]
    descriptor_file = os.path.join(output_dir, base + "_CHB_descriptors.npy")
    structures, descriptors = get_structures_and_descriptors(traj_file, descriptor_file)
    process_structures(structures, descriptors, base, output_dir, variance_threshold)

def process_all_traj_files(input_dir, variance_threshold=0.95, output_dir=".", ext="*.traj"):
    """
    Processes each .traj file in a directory separately.
    
    Parameters:
      input_dir (str): Directory containing the .traj files.
      variance_threshold (float): Variance threshold for PCA analysis.
      output_dir (str): Directory where output files will be saved.
      ext (str): File extension pattern (default: "*.traj").
    """
    traj_files = glob.glob(os.path.join(input_dir, ext))
    if not traj_files:
        print(f"No {ext} files found in {input_dir}.")
        return
    for traj_file in traj_files:
        process_traj_file(traj_file, variance_threshold=variance_threshold, output_dir=output_dir)

def get_descriptor_full_path(traj_file, output_path):
    """
    Constructs the descriptor file name based on the trajectory file path and the output directory.
    
    The descriptor file name is built using:
      1. The orientation extracted from the 6th component of the traj_file path 
         (e.g., "AIMD_oriented_incidence_noSpin" → "oriented").
      2. The prefix "continued" if the 7th component starts with "continue".
      3. The energy value extracted from the substring between "Ei." and ".Ts" 
         in the 7th component, multiplied by 1000 and appended with "meV".
      4. The run number extracted from the file name (e.g., from "vasprun-108.xml" → "108").
    
    The final descriptor file name is:
      {prefix}_{energy_meV}meV_{orientation}_{run_number}_CHB_descriptors.npy
      
    Parameters:
      traj_file (str): Full path to the trajectory file.
      output_path (str): Directory where the descriptor file should be saved.
    
    Returns:
      str: The full path to the descriptor file.
    """
    # Split the trajectory file path into its components.
    parts = traj_file.split(os.sep)
    
    # 1. Extract orientation from the folder "AIMD_oriented_incidence_noSpin"
    #    (assumed to be the 6th element in the path, i.e., parts[5])
    if len(parts) >= 6:
        aimd_folder = parts[5]
    else:
        aimd_folder = ""
    if aimd_folder.startswith("AIMD_") and "_incidence_noSpin" in aimd_folder:
        orientation = aimd_folder.split("_")[1]
    else:
        orientation = "unknown"
    
    # 2. Extract "continued" prefix from the 7th component (parts[6])
    if len(parts) >= 7:
        folder2 = parts[6]
    else:
        folder2 = ""
    prefix = ""
    if folder2.startswith("continue"):
        prefix = "continued"
    
    # 3. Extract energy between "Ei." and ".Ts", multiply by 1000, and append "meV"
    start_index = folder2.find("Ei.")
    end_index = folder2.find(".Ts")
    if start_index != -1 and end_index != -1:
        energy_str = folder2[start_index + len("Ei."):end_index]  # e.g., "0.3"
        try:
            energy_val = float(energy_str)
        except ValueError:
            energy_val = 0.0
        energy_meV = int(energy_val * 1000)  # e.g., 0.3*1000 = 300
    else:
        energy_meV = 0
    
    # 4. Extract run number from the file name "vasprun-108.xml"
    file_name = parts[-1] if parts else ""
    base_file = os.path.splitext(file_name)[0]  # e.g., "vasprun-108"
    if "-" in base_file:
        run_number = base_file.split("-")[-1]
    else:
        run_number = ""
    
    # 5. Construct the descriptor file name and join with the output directory.
    #custom_descriptor_name = f"{prefix}_{energy_meV}meV_{orientation}_{run_number}_CHB_descriptors.npy"
    custom_descriptor_name = f"{prefix}_300meV_oriented_{run_number}_CHB_descriptors.npy"
    descriptor_full_path = os.path.join(output_path, custom_descriptor_name)
    
    return descriptor_full_path

def process_all_traj_files_as_one(input_dirs, variance_threshold=0.95, output_dir=".", ext="*.traj"):
    """
    Combines all .traj files in a directory into one dataset and processes them together.
    
    Parameters:
      input_dir (str): Directory containing the .traj files.
      variance_threshold (float): Variance threshold for PCA analysis.
      output_dir (str): Directory where output files will be saved.
      ext (str): File extension pattern (default: "*.traj").
    """
    traj_files = []
    for input_dir in input_dirs:
        files = glob.glob(os.path.join(input_dir, ext))
        if not files:
            print(f"No {ext} files found in the given directory: {input_dir}")
        else:
            traj_files.extend(files)
    if not traj_files:
        return

    # Use the base name from the first file.
    #base = os.path.splitext(os.path.basename(traj_files[0]))[0]
    base = 'from_all_dinamics_' 
    all_structures  = []
    all_descriptors = []
    for traj_file in traj_files:
        descriptor_file = get_descriptor_full_path(traj_file, output_dir)
        structures, descriptors = get_structures_and_descriptors(traj_file, descriptor_file)
        all_structures.extend(structures)
        all_descriptors.extend(descriptors)
    print(f"\nTotal configurations loaded: {len(all_structures)}")
    
    #combined_descriptor_file = os.path.join(output_dir, base + "_all.npy")
    #np.save(combined_descriptor_file, all_descriptors)
    #print(f"Saved combined descriptors to {combined_descriptor_file}")
    
    process_structures(all_structures, all_descriptors, base, output_dir, variance_threshold)

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Specify the input and output directories.
    input_directory  = [#"/home/alou/Work/HOPG_O_NO/AIMD_oriented_incidence_noSpin/continue.vaspdata.Ei.0.025.Ts.300.NO.rand.zpe",
                        #                "/home/alou/Work/HOPG_O_NO/AIMD_oriented_incidence_noSpin/continue.vaspdata.Ei.0.05.Ts.300.NO.rand.zpe",
                        #                "/home/alou/Work/HOPG_O_NO/AIMD_oriented_incidence_noSpin/continue.vaspdata.Ei.0.1.Ts.300.NO.rand.zpe",
                        #                "/home/alou/Work/HOPG_O_NO/AIMD_oriented_incidence_noSpin/continue.vaspdata.Ei.0.3.Ts.300.NO.rand.zpe",
                        #                "/home/alou/Work/HOPG_O_NO/AIMD_oriented_incidence_noSpin/vaspdata.Ei.0.025.Ts.300.NO.rand.zpe",
                        #                "/home/alou/Work/HOPG_O_NO/AIMD_oriented_incidence_noSpin/vaspdata.Ei.0.05.Ts.300.NO.rand.zpe",
                        #                "/home/alou/Work/HOPG_O_NO/AIMD_oriented_incidence_noSpin/vaspdata.Ei.0.1.Ts.300.NO.rand.zpe",
                        #                "/home/alou/Work/HOPG_O_NO/AIMD_oriented_incidence_noSpin/vaspdata.Ei.0.3.Ts.300.NO.rand.zpe"]
                                        "/home/alou/Work/HOPG_O_NO/AIMD_normal_incidence_noSpin/vaspdata.Ei.0.025.Ts.300.NO.rand.zpe",
                                        "/home/alou/Work/HOPG_O_NO/AIMD_normal_incidence_noSpin/vaspdata.Ei.0.05.Ts.300.NO.rand.zpe",
                                        "/home/alou/Work/HOPG_O_NO/AIMD_normal_incidence_noSpin/vaspdata.Ei.0.1.Ts.300.NO.rand.zpe",
                                        "/home/alou/Work/HOPG_O_NO/AIMD_normal_incidence_noSpin/vaspdata.Ei.0.3.Ts.300.NO.rand.zpe"]
    output_directory = "/home/alou/Work/HOPG_O_NO/AIMD_oriented_incidence_noSpin/Chebichev_descriptors"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Set the variance threshold for PCA analysis.
    variance_threshold = 0.95

    # --- Choose one of the following modes ---
    mode = "combined"   # Options: "single", "separate", "combined"
    
    if mode == "single":
        # Process a single trajectory file.
        traj_file = os.path.join(input_directory, "NO_oriented_all_NO_react_25meV.traj")
        process_traj_file(traj_file, variance_threshold=variance_threshold, output_dir=output_directory)
    elif mode == "separate":
        # Process each .traj file in the input directory separately.
        process_all_traj_files(input_directory, variance_threshold=variance_threshold, output_dir=output_directory)
    elif mode == "combined":
        # Combine all files (using the specified extension) and process them as one dataset.
        process_all_traj_files_as_one(input_directory, variance_threshold=variance_threshold, output_dir=output_directory, ext="*.xml")
    else:
        print("Invalid mode selected. Please choose 'single', 'separate', or 'combined'.")
