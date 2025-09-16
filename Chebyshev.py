#!/usr/bin/env python
import numpy as np

# -----------------------------------------------------------------------
# Chebyshev polynomials and their derivatives
# (These functions mimic the Fortran routines in the module "chebyshev")
# -----------------------------------------------------------------------

def chebyshev_polynomial(r, r0, r1, n):
    """
    Compute Chebyshev polynomials T0, T1, ..., Tn for a given argument.
    
    The argument r is first rescaled from the interval [r0, r1] to x in [-1, 1]:
         x = (2*r - (r0 + r1)) / (r1 - r0)
    
    Parameters:
      r  : float
           The point at which to evaluate the polynomials.
      r0 : float
           Lower bound of the interval.
      r1 : float
           Upper bound of the interval.
      n  : int
           Maximum polynomial order.
    
    Returns:
      T  : numpy array of length (n+1) where T[i] = T_i(x).
    """
    x = (2.0 * r - r0 - r1) / (r1 - r0)
    T = np.zeros(n+1)
    T[0] = 1.0
    if n > 0:
        T[1] = x
        for i in range(2, n+1):
            T[i] = 2.0 * x * T[i-1] - T[i-2]
    return T

def chebyshev_polynomial_d1(r, r0, r1, n):
    """
    Compute the first derivatives of Chebyshev polynomials T0, T1, ..., Tn
    with respect to r.
    
    Parameters:
      r  : float
           The point at which to evaluate the derivatives.
      r0 : float
           Lower bound of the interval.
      r1 : float
           Upper bound of the interval.
      n  : int
           Maximum polynomial order.
    
    Returns:
      dT : numpy array of length (n+1) where dT[i] is the derivative of T_i.
    
    Note: This function is provided for completeness; in many descriptor
          applications only the function values are used.
    """
    x = (2.0 * r - r0 - r1) / (r1 - r0)
    dT = np.zeros(n+1)
    if n > 0:
        U1 = 1.0
        dT[1] = U1
        U2 = 2.0 * x
        for j in range(2, n+1):
            dT[j] = U2 * j
            U3 = 2.0 * x * U2 - U1
            U1, U2 = U2, U3
    dT *= (2.0 / (r1 - r0))
    return dT

# -----------------------------------------------------------------------
# Cutoff functions (as in sfb_fc and sfb_fc_d1)
# -----------------------------------------------------------------------

def fc(Rij, Rc):
    """
    Cutoff function.
    
    Returns 0 if Rij >= Rc; otherwise returns:
         0.5 * (cos(pi * Rij / Rc) + 1)
    
    Parameters:
      Rij : float
            Interatomic distance.
      Rc  : float
            Cutoff radius.
    """
    if Rij >= Rc:
        return 0.0
    else:
        return 0.5 * (np.cos(np.pi / Rc * Rij) + 1.0)

def fc_d1(Rij, Rc):
    """
    First derivative of the cutoff function with respect to Rij.
    
    Parameters:
      Rij : float
            Interatomic distance.
      Rc  : float
            Cutoff radius.
    """
    if Rij >= Rc:
        return 0.0
    else:
        a = np.pi / Rc
        return -0.5 * a * np.sin(a * Rij)

# -----------------------------------------------------------------------
# FingerprintBasis class (mimicking sfbasis.f90)
# -----------------------------------------------------------------------

class FingerprintBasis:
    def __init__(self, num_types, atom_types, radial_order, angular_order,
                 radial_Rc, angular_Rc):
        """
        Initialize a new structural fingerprint (descriptor) basis.
        
        Parameters:
          num_types    : Number of atomic species.
          atom_types   : List of atomic species labels (e.g., ['H', 'O']).
          radial_order : Expansion order for the radial basis.
          angular_order: Expansion order for the angular basis.
          radial_Rc    : Cutoff radius for the radial part.
          angular_Rc   : Cutoff radius for the angular part.
        """
        self.num_types  = num_types
        self.atom_types = atom_types      # List of species labels (e.g., ['H', 'O'])
        self.r_order    = radial_order
        self.a_order    = angular_order
        self.r_Rc       = radial_Rc
        self.a_Rc       = angular_Rc

        # Number of Chebyshev functions for the radial and angular parts.
        self.r_N = radial_order + 1
        self.a_N = angular_order + 1
        
        # For non-multi-component systems, the descriptor is the sum of the two parts.
        self.N = self.r_N + self.a_N
        
        # Define index ranges (0-indexed) for each block.
        # First radial block: indices [r_i1, r_f1)
        self.r_i1 = 0
        self.r_f1 = self.r_i1 + self.r_N
        # First angular block: indices [a_i1, a_f1)
        self.a_i1 = self.r_f1
        self.a_f1 = self.a_i1 + self.a_N
        # For multi-component systems, additional redundant blocks are defined:
        self.r_i2 = self.a_f1
        self.r_f2 = self.r_i2 + self.r_N
        self.a_i2 = self.r_f2
        self.a_f2 = self.a_i2 + self.a_N
        
        # If the system has more than one type, double the descriptor length.
        self.multi = (num_types > 1)
        if self.multi:
            self.N = 2 * self.N

        # Set type IDs (here simply 1, 2, ..., num_types)
        self.typeid = list(range(1, num_types + 1))
        # Compute typespin values (used for weighting in multi-component systems).
        s = - (num_types // 2)
        self.typespin = []
        for i in range(num_types):
            # Adjust for even number of species to avoid a zero spin.
            if s == 0 and (num_types % 2 == 0):
                s += 1
            self.typespin.append(float(s))
            s += 1
        
        self.initialized = True

    def print_info(self):
        """Print information about the fingerprint basis."""
        print("Radial cutoff  :", self.r_Rc)
        print("Angular cutoff :", self.a_Rc)
        print("Radial order   :", self.r_order)
        print("Angular order  :", self.a_order)
        print("Atom types     :", self.atom_types)
        print("Total number of basis functions:", self.N)
    
    # -----------------------------------------------------
    # Full implementation of the radial basis evaluation
    # -----------------------------------------------------
    def sfb_radial(self, R_ij, d_ij, compute_deriv=False):
        """
        Evaluate the radial basis functions for a given neighbor.
        
        Parameters:
          R_ij       : numpy array of shape (3,) representing the displacement vector.
          d_ij       : float, the norm (distance) of R_ij.
          compute_deriv: Boolean flag; if True, also return the derivatives.
          
        Returns:
          If compute_deriv is False:
              values: numpy array of length self.r_N (weighted Chebyshev values).
          If compute_deriv is True:
              (values, deriv_i, deriv_j), where:
                deriv_i: Derivatives (shape (3, self.r_N)) with respect to the central atom.
                deriv_j: Derivatives (shape (3, self.r_N)) with respect to the neighbor.
        """
        # Calculate the cutoff weight for the distance.
        w_ij = fc(d_ij, self.r_Rc)
        # Evaluate the Chebyshev polynomials at the distance.
        f = chebyshev_polynomial(d_ij, 0.0, self.r_Rc, self.r_order)
        values = w_ij * f

        if not compute_deriv:
            return values
        else:
            # Compute derivatives of the cutoff and Chebyshev functions.
            dw_ij = fc_d1(d_ij, self.r_Rc)
            df = chebyshev_polynomial_d1(d_ij, 0.0, self.r_Rc, self.r_order)
            # Derivative with respect to the central atom's coordinates.
            # The derivative is given by: - (R_ij/d_ij) * (dw_ij*f + w_ij*df)
            deriv_i = - (R_ij / d_ij)[:, np.newaxis] * (dw_ij * f + w_ij * df)
            # Derivative with respect to the neighbor is the negative of the above.
            deriv_j = - deriv_i
            return values, deriv_i, deriv_j

    # -----------------------------------------------------
    # Full implementation of the angular basis evaluation
    # -----------------------------------------------------
    def sfb_angular(self, R_ij, R_ik, d_ij, d_ik, cos_ijk, compute_deriv=False):
        """
        Evaluate the angular basis functions for a pair of neighbors.
        
        Parameters:
          R_ij, R_ik  : numpy arrays of shape (3,) for the displacement vectors
                        from the central atom to neighbors j and k.
          d_ij, d_ik  : floats, the norms (distances) of R_ij and R_ik.
          cos_ijk     : float, cosine of the angle between R_ij and R_ik.
          compute_deriv: Boolean flag; if True, also return the derivatives.
          
        Returns:
          If compute_deriv is False:
              values: numpy array of length self.a_N (weighted Chebyshev values).
          If compute_deriv is True:
              (values, deriv_i, deriv_j, deriv_k), where:
                deriv_i: Derivatives (shape (3, self.a_N)) with respect to the central atom.
                deriv_j: Derivatives (shape (3, self.a_N)) with respect to neighbor j.
                deriv_k: Derivatives (shape (3, self.a_N)) with respect to neighbor k.
        """
        # Compute cutoff weights for each neighbor.
        fc_j = fc(d_ij, self.a_Rc)
        fc_k = fc(d_ik, self.a_Rc)
        # Combined angular cutoff weight.
        w_ijk = fc_j * fc_k
        # Evaluate Chebyshev polynomials for the cosine value.
        f = chebyshev_polynomial(cos_ijk, -1.0, 1.0, self.a_order)
        values = w_ijk * f

        if not compute_deriv:
            return values
        else:
            # Compute derivatives of the cutoff functions.
            dfc_j = fc_d1(d_ij, self.a_Rc)
            dfc_k = fc_d1(d_ik, self.a_Rc)
            # Derivative of the Chebyshev polynomials with respect to cos(theta).
            df = chebyshev_polynomial_d1(cos_ijk, -1.0, 1.0, self.a_order)

            # Precompute inverse distance factors.
            id_ij2 = 1.0 / (d_ij ** 2)
            id_ik2 = 1.0 / (d_ik ** 2)
            id_ij_ik = 1.0 / (d_ij * d_ik)

            # Compute derivatives of cos(theta) with respect to displacements.
            # For the central atom:
            di_cos = cos_ijk * (R_ij * id_ij2 + R_ik * id_ik2) - (R_ij + R_ik) * id_ij_ik
            # For neighbor j:
            dj_cos = - cos_ijk * (R_ij * id_ij2) + R_ik * id_ij_ik
            # For neighbor k:
            dk_cos = - cos_ijk * (R_ik * id_ik2) + R_ij * id_ij_ik

            # Compute derivatives of the combined cutoff weight.
            di_w = - (dfc_j * fc_k * R_ij / d_ij + fc_j * dfc_k * R_ik / d_ik)
            dj_w =  dfc_j * fc_k * R_ij / d_ij
            dk_w =  fc_j * dfc_k * R_ik / d_ik

            # Combine the derivatives:
            # d(values)/dR = (d(w_ijk)/dR)*f + w_ijk*(df/d(cos))* d(cos)/dR.
            deriv_i = di_w[:, np.newaxis] * f[np.newaxis, :] + w_ijk * (df[np.newaxis, :] * di_cos[:, np.newaxis])
            deriv_j = dj_w[:, np.newaxis] * f[np.newaxis, :] + w_ijk * (df[np.newaxis, :] * dj_cos[:, np.newaxis])
            deriv_k = dk_w[:, np.newaxis] * f[np.newaxis, :] + w_ijk * (df[np.newaxis, :] * dk_cos[:, np.newaxis])
            return values, deriv_i, deriv_j, deriv_k

    def eval(self, coo0, itype1, coo1):
        """
        Evaluate the full structural fingerprint (descriptor) for a central atom.
        
        The descriptor is constructed by summing contributions from radial
        and angular basis functions computed over the local atomic environment.
        
        Parameters:
          coo0   : numpy array of shape (3,) with the central atom coordinates.
          itype1 : list or 1D numpy array with neighbor type indices (0-indexed).
          coo1   : numpy array of shape (nat, 3) with coordinates of nat neighbors.
          
        Returns:
          values : numpy array of shape (self.N,) containing the complete descriptor.
        """
        EPS = 1e-12  # A small number to avoid division by zero.
        nat = coo1.shape[0]
        values = np.zeros(self.N)
        
        # Loop over each neighbor j.
        for j in range(nat):
            R_ij = coo1[j] - coo0
            d_ij = np.linalg.norm(R_ij)
            if d_ij <= self.r_Rc and d_ij > EPS:
                # Evaluate the radial basis functions for neighbor j.
                f_rad = self.sfb_radial(R_ij, d_ij)
                # Add the contribution to the first radial block.
                values[self.r_i1:self.r_f1] += f_rad
                # If the system is multi-component, add a redundant block weighted
                # by the neighbor's species (typespin).
                if self.multi:
                    s_j = self.typespin[itype1[j]]
                    values[self.r_i2:self.r_f2] += s_j * f_rad

                # Only consider angular contributions if d_ij is within the angular cutoff.
                if d_ij > self.a_Rc:
                    continue

                # Loop over pairs: for each neighbor k > j.
                for k in range(j+1, nat):
                    R_ik = coo1[k] - coo0
                    d_ik = np.linalg.norm(R_ik)
                    if d_ik > self.a_Rc or d_ik < EPS:
                        continue
                    cos_ijk = np.dot(R_ij, R_ik) / (d_ij * d_ik)
                    f_ang = self.sfb_angular(R_ij, R_ik, d_ij, d_ik, cos_ijk)
                    # Add the angular contribution to the first angular block.
                    values[self.a_i1:self.a_f1] += f_ang
                    if self.multi:
                        s_j = self.typespin[itype1[j]]
                        s_k = self.typespin[itype1[k]]
                        values[self.a_i2:self.a_f2] += (s_j * s_k) * f_ang
        return values
    
    def eval_with_deriv(self, coo0, itype1, coo1):
        """
        Evaluate the full structural fingerprint (descriptor) for a central atom
        along with its derivative with respect to the central atom's coordinates.
    
        The descriptor is constructed by summing contributions from radial
        and angular basis functions computed over the local atomic environment.
    
        Parameters:
        coo0   : numpy array of shape (3,) with the central atom coordinates.
        itype1 : list or 1D numpy array with neighbor type indices (0-indexed).
        coo1   : numpy array of shape (nat, 3) with coordinates of nat neighbors.
      
        Returns:
        values : numpy array of shape (self.N,) containing the complete descriptor.
        deriv  : numpy array of shape (3, self.N) containing the derivative with respect to coo0.
        """
        EPS = 1e-12
        nat = coo1.shape[0]
        values = np.zeros(self.N)
        deriv = np.zeros((3, self.N))
    
        # Loop over each neighbor j.
        for j in range(nat):
            R_ij = coo1[j] - coo0
            d_ij = np.linalg.norm(R_ij)
            if d_ij <= self.r_Rc and d_ij > EPS:
                # Evaluate the radial basis functions and their derivatives for neighbor j.
                f_rad, deriv_i_rad, deriv_j_rad = self.sfb_radial(R_ij, d_ij, compute_deriv=True)
                # Add the contribution to the first radial block.
                values[self.r_i1:self.r_f1] += f_rad
                deriv[:, self.r_i1:self.r_f1] += deriv_i_rad
                # For multi-component systems, add the weighted redundant radial block.
                if self.multi:
                    s_j = self.typespin[itype1[j]]
                    values[self.r_i2:self.r_f2] += s_j * f_rad
                    deriv[:, self.r_i2:self.r_f2] += s_j * deriv_i_rad

                # Only consider angular contributions if d_ij is within the angular cutoff.
                if d_ij > self.a_Rc:
                    continue

                # Loop over pairs: for each neighbor k > j.
                for k in range(j+1, nat):
                    R_ik = coo1[k] - coo0
                    d_ik = np.linalg.norm(R_ik)
                    if d_ik > self.a_Rc or d_ik < EPS:
                        continue
                    cos_ijk = np.dot(R_ij, R_ik) / (d_ij * d_ik)
                    # Evaluate the angular basis functions and their derivatives.
                    f_ang, deriv_i_ang, deriv_j_ang, deriv_k_ang = self.sfb_angular(
                        R_ij, R_ik, d_ij, d_ik, cos_ijk, compute_deriv=True)
                    # Add the angular contribution to the first angular block.
                    values[self.a_i1:self.a_f1] += f_ang
                    deriv[:, self.a_i1:self.a_f1] += deriv_i_ang
                    # For multi-component systems, add the weighted redundant angular block.
                    if self.multi:
                        s_j = self.typespin[itype1[j]]
                        s_k = self.typespin[itype1[k]]
                        values[self.a_i2:self.a_f2] += (s_j * s_k) * f_ang
                        deriv[:, self.a_i2:self.a_f2] += (s_j * s_k) * deriv_i_ang
        return values, deriv
    
    # -------------------------------------------------
    # Reconstruction of the radial distribution
    # -------------------------------------------------
    def reconstruct_radial(self, coeff, nx):
        """
        Reconstruct the radial distribution function from its expansion coefficients.
        
        Parameters:
          coeff : 1D array of length self.r_N (i.e. radial_order+1) containing the expansion coefficients.
          nx    : Number of grid points for evaluation.
        
        Returns:
          x : 1D numpy array of grid points from 0 to r_Rc.
          y : 1D numpy array of the reconstructed function values.
        
        The reconstruction uses the Chebyshev polynomials on [0, r_Rc] along with
        a weight function to account for orthogonality.
        """
        PI = np.pi
        PI_INV = 1.0 / PI

        # Create a grid from 0 to r_Rc.
        x = np.linspace(0, self.r_Rc, nx)
        y = np.zeros_like(x)

        # Loop over grid points (except the last one).
        for ix in range(nx - 1):
            # Evaluate Chebyshev polynomials T0 ... T_{r_order} at x[ix].
            f = chebyshev_polynomial(x[ix], 0.0, self.r_Rc, self.r_order)
            r_over_Rc = x[ix] / self.r_Rc

            # Compute a weight to restore orthogonality.
            eps = 1e-12
            denominator = r_over_Rc - r_over_Rc**2
            if denominator < eps:
                w = 0.0
            else:
                w = 0.5 / np.sqrt(denominator)

            # Multiply the Chebyshev values by the weight and normalize.
            f = f * w * PI_INV
            # Adjust the first term (as is customary in Chebyshev expansions).
            f[0] = 0.5 * f[0]
            # Sum the contributions from all basis functions.
            y[ix] = np.sum(coeff[:self.r_N] * f[:self.r_N])
        # Set the last grid point value to zero (due to singularity at the boundary).
        y[-1] = 0.0
        return x, y

    # -------------------------------------------------
    # Reconstruction of the angular distribution
    # -------------------------------------------------
    def reconstruct_angular(self, coeff, nx):
        """
        Reconstruct the angular distribution function from its expansion coefficients.
        
        Parameters:
          coeff : 1D array of length self.a_N (i.e. angular_order+1) containing the expansion coefficients.
          nx    : Number of grid points for evaluation.
        
        Returns:
          x : 1D numpy array of grid points from 0 to pi.
          y : 1D numpy array of the reconstructed function values.
        
        The reconstruction uses the Chebyshev polynomials on [0, pi] along with a weight function.
        """
        PI = np.pi
        PI_INV = 1.0 / PI

        # Create a grid from 0 to pi.
        x = np.linspace(0, PI, nx)
        y = np.zeros_like(x)

        for ix in range(nx - 1):
            # Evaluate Chebyshev polynomials on [0, pi].
            f = chebyshev_polynomial(x[ix], 0.0, PI, self.a_order)
            r_over_PI = x[ix] / PI
            eps = 1e-12
            denominator = r_over_PI - r_over_PI**2
            if denominator < eps:
                w = 0.0
            else:
                w = 0.5 / np.sqrt(denominator)
            f = f * w * PI_INV
            f[0] = 0.5 * f[0]
            # Sum the contributions from the basis functions.
            y[ix] = np.sum(coeff[:self.a_N] * f[:self.a_N])
        y[-1] = 0.0
        return x, y