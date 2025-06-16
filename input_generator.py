import math
import itertools
import numpy as np
import time
from tqdm import tqdm
import os


class CHUCK3InputBuilder:
    def __init__(self, alpha="CHUCK3 Input"):
        self.icon = [1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0]
        self.alpha = alpha[:60].ljust(60)
        self.lines = []

        # Defaults for CARD SET 1 and 2
        self.thetan = 150.0
        self.theta1 = 1.0
        self.dtheta = 0.5

        # CARD SET 3: partial waves and channel info
        self.lmax = 40
        self.nchan = None  # Will be set later
        self.j_vals = []
        self.k_vals = []

        # CARD SET 4: integration parameters
        self.dr = 0.1
        self.rmax = 20.0

        self.channel_info = {}

        self.channel_lines = []
        self.coupling_lines = []

        self.input_card = ""

    # Card Set 1
    def set_header(self, read: bool = True,
                   write_scattering_amplitudes: bool = False,
                   print_elastic_T_matrix_elements: bool = False,
                   print_all_T_matrix_elements: bool = False,
                   print_scattering_amplitudes: bool = False,
                   differential_cross_section_log_scale: int = 3,
                   use_relativistic_kinematics: bool = False,
                   print_diagonal_channel_functions: bool = False,
                   print_off_diagonal_channel_functions: bool = False,
                   id: str = "CHUCK3 Input"):
        """
        Set ICON(12) flags and ALPHA name string.
        """
        icon = [0] * 12
        icon[0] = 1 if read else 9
        icon[4] = 1 if write_scattering_amplitudes else 0
        icon[5] = 1 if print_elastic_T_matrix_elements else 0
        icon[6] = 1 if print_all_T_matrix_elements else 0
        icon[7] = 1 if print_scattering_amplitudes else 0
        icon[8] = differential_cross_section_log_scale  # default is 3
        icon[9] = 1 if use_relativistic_kinematics else 0
        icon[10] = 1 if print_diagonal_channel_functions else 0
        icon[11] = 1 if print_off_diagonal_channel_functions else 0

        self.icon = icon
        self.alpha = id[:60].ljust(60)

    # Card Set 2
    def set_angle_distribution(self, thetan: float, theta1: float, dtheta: float):
        """
        Add CARD SET 2: angular distribution sampling settings.
        Format: 3F8.4 → must be strictly signed, 8-wide fields.
        """
        self.thetan = thetan
        self.theta1 = theta1
        self.dtheta = dtheta

    # Card Set 3
    def set_card3_partial_wave_structure(self, lmax: int, nchan: int = None):
        self.lmax = lmax
        self.nchan = nchan
        self.j_vals = []
        self.k_vals = []

    def set_card3_jk(self, j_vals: list[int], k_vals: list[int]):
        if len(j_vals) != len(k_vals):
            raise ValueError("Length of J and K lists must match")
        self.j_vals = j_vals
        self.k_vals = k_vals
        self.nchan = len(j_vals)

    def _build_card3(self):
        if self.nchan is None or not self.j_vals or not self.k_vals:
            return None  # Incomplete — skip for now
        # values = [self.lmax, self.nchan] + self.j_vals + self.k_vals
        values = [self.lmax, self.nchan] + self.j_vals

        return "".join(f"{v:+03d}" for v in values)
            
    # Card Set 4
    def set_integration_parameters(self, dr: float = 0.1, rmax: float = 20.0):
        """
        Set integration step size (DR) and radial limit (±RMAX).
        Format: 2F8.4 → signed, 8-wide, 4 decimal places
        """
        self.dr = dr
        self.rmax = rmax

        if self.rmax / self.dr > 200:
            raise ValueError("RMAX/DR must not exceed 200.0")

    def add_channel(self, 
                id: int,
                target_spin: float,
                target_parity: float,
                energy: float,
                projectile_mass: float,
                projectile_charge: float,
                projectile_spin: float,
                target_mass: float,
                target_z: float,
                coulomb_charge_diffuseness: float,
                nonlocal_range_parameter: float,
                optical_model_parameters: str = "",
                q_value: float = None,
                excited_state_energy: float = None,
                ldfrm = None,
                beta = None,
                beta2 = None,
                beta4 = None,
            ):
        
        J_channel = int(round(2 * target_parity * target_spin))
        K_channel = 0

        omp_lines = []
        if optical_model_parameters == "":
            # error if no optical model parameters are provided
            raise ValueError("Optical model parameters must be provided")
        elif optical_model_parameters == "Koning-Delaroche":
            # Use Koning-Delaroche 2009 model for protons
            lines, r_coulomb = self.koning_delariche_2009_protons(energy, target_z, target_mass)
            omp_lines.extend(lines)
        elif optical_model_parameters == "Becchetti-Greenless":
            # Use Becchetti-Greenless model for tritons
            lines, r_coulomb = self.becchetti_triton_potential(energy, target_z, target_mass)
            omp_lines.extend(lines)

        def format_smart_float(value: float) -> str:
            """
            Format a float with adaptive width:
            - If abs(value) < 100 → use +07.3f
            - If abs(value) >= 100 → use +08.3f
            """
            if abs(value) < 100:
                return f"{value:+07.3f}"
            else:
                return f"{value:+08.3f}"

        # Card 1: 9F8.4
        card1 = (
            f"{(energy if q_value is None else q_value):+08.4f}"
            f"{projectile_mass:+08.4f}"
            f"{projectile_charge:+08.4f}"
            f"{format_smart_float(target_mass)}"
            f"{target_z:+08.4f}"
            f"{r_coulomb:+08.4f}"
            f"{coulomb_charge_diffuseness:+08.4f}"
            f"{nonlocal_range_parameter:+08.4f}"
            f"{2 * projectile_spin:+08.4f}"
        )
        # Append excited state energy if provided
        if excited_state_energy is not None:
            card1 += f"{-excited_state_energy:+08.4f}"
            
        self.channel_lines.append(card1)

        # ---- CARD 2 ---- (always IDs, optional deformation)
        card2 = (
            f"{id:+03d}{id:+03d}"
        )
        # I do not use, so unsure if this is 100% correct
        if None not in (ldfrm, beta, beta2, beta4):
            card2 += (
                " " * 12 +
                f"{ldfrm:3d}" + " " * 3 +
                f"{beta:+08.4f}"
                f"{beta2:+08.4f}"
                f"{beta4:+08.4f}"
            )

        self.channel_lines.append(card2)

        # Add optical model parameters
        for line in omp_lines:
            self.channel_lines.append(line)

        self.channel_info[id] = {
            "target_spin": target_spin,
            "target_parity": target_parity,
            "energy": energy,
            "q_value": q_value,
            "excited_state_energy": excited_state_energy,
            "projectile_spin": projectile_spin,
            "target_mass": target_mass,
            "target_z": target_z,
            "J_channel": J_channel,
            "optical_model_lines": omp_lines,
            "r_c": r_coulomb,
        }

        # CARD SET 3 update
        self.j_vals.append(J_channel)
        self.k_vals.append(K_channel)
        self.nchan = len(self.j_vals)

        if self.nchan > 8:
            raise ValueError("Number of channels (nchan) must not exceed 8")

    def optical_model_parameters(self,
                                 option: int,
                                 v_r: float = None, # Real well depth
                                 r_or: float = None, # Real well radius
                                 a_r: float = None, # Real well diffuseness
                                 v_sor: float = None, # Thomas spin-orbit factor
                                 v_i: float = None, # Imaginary well depth
                                 r_oi: float = None, # Imaginary well radius
                                 a_i: float = None, # Imaginary well diffuseness
                                 powr: float = None, # Power for the imaginary well
                                 ): 
        
        """
        Add Cards 3-M: Optical model parameters.
        FORMAT (8F8.4, 8X, F8.4)
        Each field is either formatted as +XXXX.XXXX or blank (8 spaces).
        POWR is always last, after 8-space pad.
        """

        def fmt(val):
            if val is None:
                return " " * 8
            elif abs(val) >= 100:
                return f"{val:+08.3f}"  # one less decimal
            else:
                return f"{val:+08.4f}"  # full precision
        
        # Apply corrections based on option
        if option in (1, -1):
            v_r = -v_r if v_r is not None else None
            v_i = -v_i if v_i is not None else None
        elif option in (2, -2):
            v_i = 4.0 * v_i if v_i is not None else None
        elif option in (4, -4):
            v_r = -4.0 * v_r if v_r is not None else None


        fields = [
            fmt(option),
            fmt(v_r),
            fmt(r_or),
            fmt(a_r),
            fmt(v_sor),
            fmt(v_i),
            fmt(r_oi),
            fmt(a_i)
        ]

        lines = []
        line = "".join(fields) + " " * 8 + (f"{powr:+08.4f}" if powr is not None else "")

        lines.append(line)
        return lines
        
    def koning_delariche_2009_protons(self, E: float, zt: int, at: int):
        """Koning-Delaroche proton scattering optical model potential

        From  Koning, A.J., Delaroche, J.P., "Local and global nucleon optical models from 1 keV to 200 MeV", Nuclear Physics A, 713, 2003
        https://doi.org/10.1016/S0375-9474(02)01321-0

        Parameters
        ----------
        E: float
            The projectile energy in MeV
        zt: int
            The target Z
        at: int
            The target A
        params: dict[str, float]
            The dictionary of optical model parameters to be filled out
        """
        nt = at - zt
        a3 = math.pow(at, 1.0 / 3.0)

        v1 = 59.30 + 21.0 * (nt - zt) / at - 0.024 * at
        v2 = 0.007067 + 4.23e-6 * at
        v3 = 1.729e-5 + 1.136e-8 * at
        v4 = 7.0e-9

        w1 = 14.667 + 0.009629 * at
        w2 = 73.55 + 0.0795 * at

        d1 = 16.0 + 16.0 * (nt - zt) / at
        d2 = 0.0180 + 0.003802 / (1.0 + math.exp((at - 156.0) / 8.0))
        d3 = 11.5

        vso1 = 5.922 + 0.0030 * at
        vso2 = 0.0040

        wso1 = -3.1
        wso2 = 160

        ef = -8.4075 + 0.01378 * at
        rc = 1.198 + 0.697 * at ** (-0.666) + 12.994 * at ** (-1.666)
        vc = 1.73 / rc * zt / a3

        delta_e = E - ef
        V = v1 * (
            1.0 - v2 * delta_e + v3 * delta_e**2.0 - v4 * delta_e**3.0
        ) + vc * v1 * (v2 - 2.0 * v3 * delta_e + 3.0 * v4 * delta_e**2.0)

        Vi = w1 * delta_e**2.0 / (delta_e**2.0 + w2**2.0)
        Vsi = (
            d1 * delta_e**2.0 / (delta_e**2.0 + d3**2.0) * math.exp(-1.0 * d2 * delta_e)
        )
        Vso = vso1 * math.exp(-1.0 * vso2 * delta_e)
        Vsoi = wso1 * delta_e**2.0 / (delta_e**2.0 + wso2**2.0)
        r0 = 1.3039 - 0.4054 / a3
        ri0 = r0
        rsi0 = 1.3424 - 0.01585 * a3
        rso0 = 1.1854 - 0.647 / a3
        rsoi0 = rso0
        rc0 = rc
        a = 0.6778 - 1.487e-4 * at
        ai = 0.6778 - 1.487e-4 * at
        asi = 0.5187 + 5.205e-4 * at
        aso = 0.59
        asoi = 0.59

        # Build the optical model parameters lines
        lines = []

        # Volume (Option 1)
        volume = self.optical_model_parameters(
            option=1,
            v_r=V,
            r_or=r0,
            a_r=a,
            v_sor=None,
            v_i=Vi,
            r_oi=ri0,
            a_i=ai,
            powr=None
        )

        lines.extend(volume)

        # Surface (Option 2)
        surface = self.optical_model_parameters(
            option=2,
            v_r=None,
            r_or=None,
            a_r=None,
            v_sor=None,
            v_i=Vsi,
            r_oi=rsi0,
            a_i=asi,
            powr=None
        )

        lines.extend(surface)

        # Spin-Orbit (Option 3)
        spin_orbit = self.optical_model_parameters(
            option=-4,
            v_r=Vso,
            r_or=rso0,
            a_r=aso,
            v_sor=None,
            v_i=None,
            r_oi=None,
            a_i=None,
            powr=None
        )

        lines.extend(spin_orbit)

        return lines, rc

    def becchetti_triton_potential(self, E: float, zt: int, at: int):
        '''
        Becchetti and Greenless, (1971) E < 40 | 40 < A | Iso. Dep.
        '''

        N = at - zt

        v = 165.0 - 0.17 * E - 6.4 * (N - zt) / at

        r0 = 1.20
        a = 0.72

        vi = 46.0 - 0.33 * E - 110.0 * (N - zt) / at
        ri0 = 1.4
        ai = 0.84

        vsi = 0.0
        rsi0 = 0.0
        asi = 0.0

        vso = 2.5
        rso0 = 1.2
        aso = 0.72

        rc = 1.300

        # Build the optical model parameters lines
        lines = []

        volume = self.optical_model_parameters(
            option=1,
            v_r=v,
            r_or=r0,
            a_r=a,
            v_sor=None,
            v_i=vi,
            r_oi=ri0,
            a_i=ai,
            powr=None
        )

        lines.extend(volume)

        surface = self.optical_model_parameters(
            option=2,
            v_r=None,
            r_or=None,
            a_r=None,
            v_sor=None,
            v_i=vsi,
            r_oi=rsi0,
            a_i=asi,
            powr=None
        )
        lines.extend(surface)

        spin_orbit = self.optical_model_parameters(
            option=-4,
            v_r=vso,
            r_or=rso0,
            a_r=aso,
            v_sor=None,
            v_i=None,
            r_oi=None,
            a_i=None,
            powr=None
        )
        lines.extend(spin_orbit)
        return lines, rc
  
    # Card Set 5
    def add_coupling(self,
                    channel_id_in: int,
                    channel_id_out: int,
                    l: int,
                    s: float,
                    icode: int,
                    ldfrm: int = 0,
                    coupling_one_way: bool = True,
                    icouex: bool = False,
                    beta: float = None,
                    beta2: float = None,
                    beta4: float = None,
                    radius_for_coulomb_excitation: float = None,
                    finite_range_parameter: float = None,

                    cntrl: float = None,
                    qcode: float = None,
                    flmu: float = None,
                    vzero: float = None,
                    fj2: float = None,
                    fji: float = None,
                    fjf: float = None,

                    E: float = None,
                    mp: float = None,
                    zp: float = None,
                    mt: float = None,
                    zt: float = None,
                    roc: float = None,
                    E_excited_state: float = None,

                    omp_lines: list[str] = None,

                    N: float = None,  
                    L: float = None,
                    J: float = None,
                    S: float = None,
                    VTRIAL: float = None,
                    FISW: float = None,
                    DAMP: float = None,

                    ):
        """
        Create a CHUCK3-compliant coupling line with fixed-format fields.
        FORMAT (8I3, 5F8.4)
        """

        if icode not in (0, 1, 2):
            raise ValueError("ICODE must be 0, 1, or 2. 0 for collective, 1 for single-particle transfer, 2 for two-particle transfer")
        
        def fmt_int(val: int) -> str:
            return f"{val:+03d}"

        def fmt_float(val: float) -> str:
            return f"{val:+08.4f}" if val is not None else " " * 8
        
        def fmt_f8(val):
            if val is None:
                return " " * 8
            elif abs(val) >= 100:
                return f"{val:+08.3f}"
            else:
                return f"{val:+08.4f}"
            
        def fmt_fixed_width_float(val: float) -> str:
            """
            Format a float into exactly 8 characters:
            - Uses fewer decimal places as the integer part grows.
            - Keeps sign and decimal point.
            """
            if val is None:
                return " " * 8

            abs_val = abs(val)
            if abs_val < 10:
                return f"{val:+08.4f}"
            elif abs_val < 100:
                return f"{val:+08.3f}"
            elif abs_val < 1000:
                return f"{val:+08.2f}"
            elif abs_val < 10000:
                return f"{val:+08.1f}"
            else:
                return f"{val:+08.0f}"

        # Handle directionality
        nj = -channel_id_out if coupling_one_way else channel_id_out
        ni = -channel_id_in if coupling_one_way else channel_id_in


        # Use channel_id_out to extract J
        channel_info = self.channel_info.get(channel_id_out)
        if channel_info is None:
            raise ValueError(f"No channel info found for channel ID {channel_id_out}")

        Jf_2 = channel_info.get("J_channel")
        if Jf_2 is None:
            raise ValueError(f"No J value found in channel info for channel ID {channel_id_out}")
        
        # get r_c from channel info
        r_c = channel_info.get("r_c")

        if radius_for_coulomb_excitation is None:
            radius_for_coulomb_excitation = channel_info.get("r_c")

        # 8 signed I3 values
        ints = [
            fmt_int(nj),
            fmt_int(ni),
            fmt_int(l),
            fmt_int(int(2 * s)),
            fmt_int(int(2*(s+l))),
            fmt_int(icode),
            fmt_int(ldfrm),
            fmt_int(1 if icouex else 0)
        ]
        int_block = "".join(ints)

        # 5 signed F8.4 values (or blank if None)
        floats = [
            fmt_fixed_width_float(beta),
            fmt_fixed_width_float(beta2),
            fmt_fixed_width_float(beta4),
            fmt_float(r_c),
            fmt_float(finite_range_parameter)
        ]
        float_block = "".join(floats)

        # Combine and append
        line = int_block + float_block
        self.coupling_lines.append(line)

        # Collective coupling (ICODE 0)
        if icode == 0:
            # get the optical model parameters from the channel info
            channel_info = self.channel_info.get(channel_id_in, {})
            omp_lines = channel_info.get("optical_model_lines", [])
            if not omp_lines:
                raise ValueError(f"No optical model parameters found for channel {channel_id_in}")
            # Need to increase the option values by 1 or -1
            for omp_line in omp_lines:
                if len(omp_line) >= 8:
                    try:
                        option_str = omp_line[:8].strip()
                        option_val = int(float(option_str))
                        # Apply transformation
                        if option_val > 0:
                            option_val += 1
                        elif option_val < 0:
                            option_val -= 1
                        new_option_str = f"{option_val:+08.4f}"
                        modified_line = new_option_str + omp_line[8:]
                        self.coupling_lines.append(modified_line)
                    except ValueError:
                        # Fallback to unmodified if parsing fails
                        self.coupling_lines.append(omp_line)
                else:
                    self.coupling_lines.append(omp_line)
        elif icode == 1:
            raise NotImplementedError("ICODE 1 (single-particle transfer) is not implemented yet")
        elif icode == 2:
            # Microscopic transfer form factors (ICODE = 2)

            # Required ICODE=2 basics
            if cntrl is None:
                raise ValueError(
                    "cntrl must be provided for ICODE = 2 (microscopic transfer form factor)\n"
                    "  0.0 = Read zero sets of orbital cards and exit\n"
                    "  1.0 = Read one set of orbital cards\n"
                    "  2.0 = Read two sets of orbital cards"
                )
            if qcode is None:
                raise ValueError(
                    "qcode must be provided for ICODE = 2. Options:\n"
                    "  0.0 = No option\n"
                    "  1.0 = Yukawa potential microscopic form factor\n"
                    "  2.0 = Coulomb potential microscopic form factor\n"
                    "  3.0 = Tensor potential microscopic form factor\n"
                    "  5.0 = Two nucleon transfer microscopic form factor\n"
                    "  6.0 = Zero range knockout microscopic form factor"
                )

            # Validate specific field requirements by QCODE
            if qcode in (1.0, 3.0) and flmu is None:
                raise ValueError("flmu (range^-1 of potentials) must be provided for QCODE = 1.0 or 3.0.")
            if qcode == 5.0 and flmu is None:
                flmu = 0.0  # Will use CHUCK3's default: 1.7
            if qcode in (1.0, 2.0, 3.0, 5.0, 6.0) and vzero is None:
                raise ValueError(
                    "vzero must be provided for QCODE = 1.0, 2.0, 3.0, 5.0, or 6.0.\n"
                    "  - Strength of potential (QCODE=1,3)\n"
                    "  - Spectroscopic amplitude (QCODE=5)\n"
                    "  - Volume integral of two-body potential (QCODE=6)"
                )
            if qcode in (1.0, 2.0, 3.0, 6.0):
                if fj2 is None:
                    raise ValueError("fj2 (2×spin of core) must be provided for QCODE = 1.0, 2.0, 3.0, or 6.0.")
                if fji is None:
                    raise ValueError("fji (2×spin of initial nucleus) must be provided for QCODE = 1.0, 2.0, 3.0, or 6.0.")
                if fjf is None:
                    raise ValueError("fjf (2×spin of final nucleus) must be provided for QCODE = 1.0, 2.0, 3.0, or 6.0.")

            # Write ICODE=2 card: FORMAT (7F8.4)
            line = (
                (f"{cntrl:+08.4f}" if cntrl is not None else " " * 8) +
                (f"{qcode:+08.4f}" if qcode is not None else " " * 8) +
                (f"{flmu:+08.4f}" if flmu is not None else " " * 8) +
                (f"{vzero:+08.4f}" if vzero is not None else " " * 8) +
                (f"{fj2:+08.4f}" if fj2 is not None else " " * 8) +
                (f"{fji:+08.4f}" if fji is not None else " " * 8) +
                (f"{fjf:+08.4f}" if fjf is not None else " " * 8)
            )
            self.coupling_lines.append(line)

            # --- Card 2 for transferred particle info (6F8.4) ---
            missing_fields = []
            if E is None:
                missing_fields.append("E (binding energy of transferred particle to core)")
            if mp is None:
                missing_fields.append("mp (mass of transferred particle)")
            if zp is None:
                missing_fields.append("zp (charge of transferred particle)")
            if mt is None:
                missing_fields.append("mt (mass of core which binds particle)")
            if zt is None:
                missing_fields.append("zt (charge of binding core, less that of transferred particle)")
            # Optional fields, can be None
            # if roc is None:
            #     missing_fields.append("roc (Coulomb radius parameter, Rc = Roc × MT^(1/3))")
            # if E_excited_state is None:
            #     missing_fields.append("E_excited_state (excitation energy (half if 2-neutron transfer) of core after transfer)")

            if missing_fields:
                raise ValueError("The following parameters must be provided for ICODE=2 Card 2:\n  - " + "\n  - ".join(missing_fields))

            # Write Card 2 line: FORMAT (6F8.4 + 3 blanks + E_excited_state in 10th spot)
            card2_line = (
                fmt_f8(-E) +
                fmt_f8(mp) +
                fmt_f8(zp) +
                fmt_f8(mt) +
                fmt_f8(zt) +
                fmt_f8(roc) +     # blanks if None
                " " * 24 +        # 3 blank fields (slots 7–9)
                fmt_f8(E_excited_state)
            )
            self.coupling_lines.append(card2_line)

            # --- Card 3-M to decribe the binding well ---
            if not omp_lines:
                raise ValueError(
                    "Binding potential well must be defined using optical model parameters.\n"
                    "Use `omp_lines=[...]` to specify the potential describing the transferred particle.\n"
                    "Typically includes a volume, surface, and spin-orbit term (e.g. Option=1, 2, ..).\n"
                    "If you say 'neutron', it will use the default neutron potential. (-01.    -01.    +01.170 +00.75  +25.)\n"
                    ""
                )
            
            if omp_lines == "neutron":
                line = self.optical_model_parameters(
                    option=-1,  # Volume term
                    v_r=1.0,  # No real well depth
                    r_or=1.170,  # Default radius
                    a_r=0.75,  # Default diffuseness
                    v_sor=25,  # No spin-orbit term
                    v_i=None,  # No imaginary well depth
                    r_oi=None,  # No imaginary radius
                    a_i=None,  # No imaginary diffuseness
                    powr=None  # No power factor
                )

                self.coupling_lines.extend(line)
            else:
                # Add each optical model parameter line
                for omp_line in omp_lines:
                    self.coupling_lines.append(omp_line)

            # --- Card M+1 for orbital info (7F8.4) ---
            m1_missing = []
            if N is None: m1_missing.append("N (number of nodes of orbital)")
            if L is None: m1_missing.append("L (orbital angular momentum)")
            if J is None: m1_missing.append("J (total angular momentum of particle (number is doubled in the output))")
            if S is None: m1_missing.append("S (intrinsic spin of particle (number is doubled in the output))")
            if VTRIAL is None: m1_missing.append("VTRIAL (scaling factor for bound state potential)")
            if FISW is not None and FISW not in (0, 1, 2):
                m1_missing.append("FISW must be 0, 1, or 2:\n"
                                "  0 = Search on well depth\n"
                                "  1 = Search on binding energy for fixed potential\n"
                                "  2 = No search")
            if DAMP is None: m1_missing.append("DAMP (exponential damping factor)")

            if m1_missing:
                raise ValueError("The following parameters must be provided for ICODE=2 Card M+1:\n  - " + "\n  - ".join(m1_missing))
            

            card_m1_line = (
                fmt_f8(N) +
                fmt_f8(L) +
                fmt_f8(2 * J) +
                fmt_f8(2 * S) +
                fmt_f8(VTRIAL) +
                fmt_f8(FISW) +
                fmt_f8(DAMP)
            )
            self.coupling_lines.append(card_m1_line)

            # Add +000 to the end of the coupling line
            self.coupling_lines[-1] += "\n+00"

    def build(self, save_path: str = None):
        icon_str = "".join(str(i) for i in self.icon)
        header_line = f"{icon_str}{' ' * 8}{self.alpha}"

        result = [header_line]

        # CARD SET 2 (angular distribution)
        angle_line = (
            f"{self.thetan:+07.3f} "
            f"{self.theta1:+07.3f} "
            f"{self.dtheta:+07.3f} "
        )

        result.append(angle_line)

        # CARD SET 3 (partial wave and channel structure)
        card3_line = self._build_card3()
        if card3_line:
            result.append(card3_line)

        # CARD SET 4
        dr_line = f"{self.dr:+08.4f}{self.rmax:+08.4f}"
        result.append(dr_line)

        # channel lines
        for line in self.channel_lines:
            result.append(line)

        for line in self.coupling_lines:
            result.append(line)


        # End-of-data marker
        result.append("+00+00\n9")

        full_text = "\n".join(result)

        self.input_card = full_text

        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(full_text)

        return full_text

    def extract_orbital(self, orbital: str):
        """
        Extract n, l, j from orbital notation string, e.g., "2f7/2".

        Returns:
            n (int): principal quantum number minus one (CHUCK3 style)
            l (int): orbital angular momentum (s=0, p=1, ..., l=9)
            j (float): total angular momentum (e.g., 7/2 → 3.5)
        """

        if len(orbital) < 3:
            raise ValueError(f"Orbital string too short: '{orbital}'")

        # Extract n and check
        try:
            n = int(orbital[0])
        except ValueError:
            raise ValueError(f"Could not parse principal quantum number from '{orbital}'")

        if n <= 0:
            raise ValueError("Orbital n must be greater than 0")

        n -= 1  # CHUCK3 uses n - 1

        # Map letter to l
        l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'j': 7, 'k': 8, 'l': 9}
        l_char = orbital[1].lower()
        if l_char not in l_map:
            raise ValueError(f"Invalid orbital letter '{l_char}' in '{orbital}'")
        l = l_map[l_char]

        # Extract and parse j
        j_str = orbital[2:].strip()
        try:
            if '/' in j_str:
                num, denom = map(int, j_str.split('/'))
                j = num / denom
            else:
                j = float(j_str)
        except Exception:
            raise ValueError(f"Invalid j value '{j_str}' in orbital '{orbital}'")

        # print(f"Extracted orbital: n={n}, l={l}, j={j} from '{orbital}'")
        return n, l, j
    
    @staticmethod
    def combine_input_cards(builders: list["CHUCK3InputBuilder"]) -> str:
        """
        Combine multiple CHUCK3 input cards into a single input.
        Strips the final "9" from all but the last builder.
        """
        combined_lines = []

        for i, builder in enumerate(builders):
            built = builder.build()
            lines = built.strip().split("\n")

            # Strip "9" from all but the last builder
            if i < len(builders) - 1:
                if lines[-1].strip() == "9":
                    lines = lines[:-1]
                else:
                    raise ValueError(f"Builder {i} does not end with '9'")

            combined_lines.extend(lines)

        return "\n".join(combined_lines)

    # def run(self, chuck_excutable_path: str, save_path: str = None):

def parse_chuck3_output_for_optimization(file_path: str, coupling_num: int = 1):
    """
    Extracts theta and sigma values from CHUCK3 output for a specific coupling.
    Returns: theta_list, sigma_list (both as floats)
    """
    results = {}
    current_name = None
    pending_name = None
    collecting = False

    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            parts = stripped.split()

            # Detect start of a coupling section
            if stripped.startswith("1") and "Sigma" in stripped and "channel" in stripped:
                if len(parts) > 7:
                    pending_name = " ".join(parts[7:-1])
                collecting = False
                continue

            if "Theta" in line and "Sigma(" in line and "Cos" in line:
                if pending_name:
                    start = line.find("Sigma(")
                    end = line.find(")", start)
                    if start != -1 and end != -1:
                        coupling = line[start + len("Sigma("):end].strip()
                    else:
                        coupling = ""
                    
                    coupling_int = int(coupling) if coupling.isdigit() else None
                    if coupling_num is not None and coupling_int != coupling_num:
                        collecting = False
                        pending_name = None
                        continue

                    current_name = f"{pending_name} (coupling: {coupling})" if coupling else pending_name
                    results[current_name] = []

                pending_name = None
                collecting = True
                continue

            if collecting and stripped.startswith("0Totl_Sig"):
                collecting = False
                continue

            if collecting and current_name:
                if len(parts) >= 2:
                    try:
                        theta = float(parts[0])
                        sigma = float(parts[1]) * 1000  # Convert from b/sr to mb/sr
                        results[current_name].append((theta, sigma))
                    except ValueError:
                        continue

    # Return the first non-empty result
    for data in results.values():
        if data:
            thetas, sigmas = zip(*data)
            return np.array(thetas), np.array(sigmas)

    # If nothing found
    return np.array([]), np.array([])

def possible_l_transfers(
    target_spin: float,
    target_parity: float,
    excited_spin: float,
    excited_parity: float
):
    J_i = target_spin
    J_f = excited_spin

    # Calculate allowed l transfer range from triangle inequality
    min_l = int(abs(J_i - J_f))
    max_l = int(J_i + J_f)

    # Generate possible l values
    possible_l = list(range(min_l, max_l + 1))

    # Apply parity selection rule
    parity_ratio = excited_parity / target_parity  # should be ±1
    allowed_l = [l for l in possible_l if (-1)**l == parity_ratio]

    # Print the result
    # print(f"Allowed ℓ-transfer values from Jπ={J_i:.1f}{'+' if target_parity > 0 else '-'} "
    #       f"to Jπ={J_f:.1f}{'+' if excited_parity > 0 else '-'}: {allowed_l}")

    return allowed_l

# Reaction: 150Nd(p,t)148Nd
energy = 16.0 # MeV

# Target: 150Nd
target_mass = 150
target_z = 60
target_spin = 0
target_parity = 1.0
target_str = "150Nd"
target_gs_spin = 0.0
target_gs_parity = 1.0  # +1 for positive parity, -1 for negative parity
beta2_target = 0.285 # deformation from NNDC

# Projectile: Proton
projectile_mass = 1.0078
projectile_spin = 1/2
projectile_charge = 1
projectile_omp = "Koning-Delaroche"
projectile_str = "p"

# Ejectile: Triton
ejectile_mass = 3.0155
ejectile_spin = 1/2
ejectile_charge = 1
ejectile_omp = "Becchetti-Greenless"
ejectile_str = "t"

# Residual: 148Nd
residual_mass = 148
residual_z = 60
residual_str = "148Nd"
residual_gs_spin = 0.0
residual_gs_parity = 1.0  # +1 for positive parity, -1 for negative parity
beta2_residual = 0.201  # deformation from NNDC
GS_orbital = "2f7/2"  # Ground state orbital of 148Nd, used in the 2-step configuration

# Q-value and neutron separation energy
Q_value = -3.932  # MeV, Q-value for the reaction
neutron_seperation_energy = 7.322  # MeV, neutron separation energy for 148Nd

chuck3_executable = "./CHUCK3-docker/run_chuck3.sh"  # Path to the CHUCK3 executable, adjust as needed

def run_chuck3_input(
        executable_path: str,
        input_file: str,
        output_file: str = None,
    ):
    """
    Run CHUCK3 with the given input file.
    """
    import subprocess

    if output_file is None:
        result = subprocess.run(
            [executable_path, input_file],
            capture_output=True,
        )
    else:
        result = subprocess.run(
            [executable_path, input_file, output_file],
            capture_output=True,
        )

    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
    
# (p,t) direct transfer
def direct_transfer(
        excited_state_spin: float, 
        excited_state_parity: float, 
        excited_state_energy: float, 
        orbitals: list[str], 
        input_path: str,
        output_path: str,
        run: bool = False
        ):
    
    l_values = possible_l_transfers(
        target_spin=target_spin,
        target_parity=target_parity,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity
    )
    
    builders = []
    for orbital in orbitals:
        parity = "plus" if excited_state_parity == 1 else "minus"
        # replace the / in orbital with _
        orbital_name = orbital.replace("/", "_")
        name = f"{excited_state_energy*1000:.0f}keV_{excited_state_spin:.0f}{parity}_{orbital_name}_direct_transfer"
        
        c = CHUCK3InputBuilder(alpha=name)
        # does not plot the cross sections
        # c.set_header(differential_cross_section_log_scale=0)

        N, L, J = c.extract_orbital(orbital)

        c.add_channel(
            id=1,
            target_spin=target_spin,
            target_parity=target_parity,
            energy=energy,
            projectile_mass=projectile_mass,
            projectile_charge=projectile_charge,
            target_mass=target_mass,
            target_z=target_z,
            coulomb_charge_diffuseness=0.0,
            nonlocal_range_parameter=0.85,
            projectile_spin=projectile_spin,
            optical_model_parameters=projectile_omp,
        )

        c.add_channel(
            id=2,
            target_spin=excited_state_spin,
            target_parity=excited_state_parity,
            energy=energy,
            q_value=Q_value,
            excited_state_energy=excited_state_energy,
            projectile_mass=ejectile_mass,
            projectile_charge=ejectile_charge,
            target_mass=residual_mass,
            target_z=residual_z,
            coulomb_charge_diffuseness=0.0,
            nonlocal_range_parameter=0.25,
            projectile_spin=ejectile_spin,
            optical_model_parameters=ejectile_omp,
        )

        for l in l_values:
            c.add_coupling(
                channel_id_in=1,
                channel_id_out=2,
                l=l,
                s=0,
                icode=2,
                ldfrm=0,
                coupling_one_way=True,
                icouex=False,
                beta=-1560.0,

                cntrl=1.0,
                qcode=5.0,
                vzero=1.0,

                E=neutron_seperation_energy,
                mp=1.008,
                zp=0.0,
                mt=residual_mass,
                zt=residual_z,
                E_excited_state=excited_state_energy/2, # Half if two-neutron transfer

                omp_lines="neutron",

                N=N,
                L=L,
                J=J,
                S=1/2,
                VTRIAL=60.0,
                FISW=0,
                DAMP=0.0,
            )

        c.build()

        builders.append(c)

    input_file = CHUCK3InputBuilder.combine_input_cards(builders)

    # adjust the save name to remove the orbital part if there are multiple orbitals
    if len(orbitals) > 1:
        name = f"{excited_state_energy*1000:.0f}keV_{excited_state_spin:.0f}{parity}_direct_transfer"

    # Check to see if the input path file already exists using os.path.exists, if so ask the user if they want to overwrite it
    if os.path.exists(input_path):
        overwrite = input(f"Input file {input_path} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Exiting without overwriting the input file.")
            return
        
    with open(input_path, 'w') as f:
        f.write(input_file)

    if run:
        print(f"Running CHUCK3 with input file: {input_path} and output file: {output_path}")

        run_chuck3_input(
            executable_path=chuck3_executable,
            input_file=input_path,
            output_file=output_path,
        )

def two_step_config_1_input(
        target_excited_state_spin: float,
        target_excited_state_parity: float,
        target_excited_state_energy: float,
        excited_state_spin: float, 
        excited_state_parity: float, 
        excited_state_energy: float,
        couplings: list[float]
    ):

    '''
    Create a CHUCK3 input for a two-step configuration with specified couplings.

    c0 = Coupling strength for the transfer from ground state in target to ground state in residual nucleus
    c1 = Coupling strength for the transfer from ground state in target to excited state in residual nucleus
    c2, c3, ... = Coupling strengths for the transfer from excited state in target to excited state in residual nucleus
    '''

    name = "_".join([f"{v:.0f}" for v in couplings])

    target_excited_state_to_residual_excited_state_l_values = possible_l_transfers(
            target_spin=target_excited_state_spin,
            target_parity=target_excited_state_parity,
            excited_spin=excited_state_spin,
            excited_parity=excited_state_parity
        )
    
    target_l_values = possible_l_transfers(
            target_spin=target_spin,
            target_parity=target_parity,
            excited_spin=target_excited_state_spin,
            excited_parity=target_excited_state_parity
        )

    c = CHUCK3InputBuilder()
    c.set_header(differential_cross_section_log_scale=3, id=name) 

    # Target GS
    c.add_channel(
        id=1,
        target_spin=target_gs_spin,
        target_parity=target_gs_parity,
        energy=energy,
        projectile_mass=projectile_mass,
        projectile_charge=projectile_charge,
        target_mass=target_mass,
        target_z=target_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.85,
        projectile_spin=projectile_spin,
        optical_model_parameters=projectile_omp,
    )

    # Target excited state
    c.add_channel(
        id=2,
        target_spin=target_excited_state_spin,
        target_parity=target_excited_state_parity,
        energy=energy,
        q_value=0.0,
        excited_state_energy=target_excited_state_energy,
        projectile_mass=projectile_mass,
        projectile_charge=projectile_charge,
        target_mass=target_mass,
        target_z=target_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.85,
        projectile_spin=projectile_spin,
        optical_model_parameters=projectile_omp,
    )

    # Coupling from GS to excited state in target
    # Collective coupling (qcode=0)
    for l in target_l_values:
        c.add_coupling(
            channel_id_in=1,
            channel_id_out=2,
            l=l,
            s=0,
            coupling_one_way=True,
            ldfrm=3,
            icouex=True,
            beta=beta2_target,
            icode=0,
        )

    # Residual nucleus GS
    c.add_channel(
        id=3,
        target_spin=residual_gs_spin,
        target_parity=residual_gs_parity,
        energy=energy,
        q_value=Q_value,
        excited_state_energy=0.0,
        projectile_mass=ejectile_mass,
        projectile_charge=ejectile_charge,
        target_mass=residual_mass,
        target_z=residual_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.25,
        projectile_spin=ejectile_spin,
        optical_model_parameters=ejectile_omp,
    )

    # Residual nucleus excited state
    c.add_channel(
        id=4,
        target_spin=excited_state_spin,
        target_parity=excited_state_parity,
        energy=energy,
        q_value=Q_value,
        excited_state_energy=excited_state_energy,
        projectile_mass=ejectile_mass,
        projectile_charge=ejectile_charge,
        target_mass=residual_mass,
        target_z=residual_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.25,
        projectile_spin=ejectile_spin,
        optical_model_parameters=ejectile_omp,
    )

    # Coupling from residual GS to excited state
    # Collective coupling (qcode=0)
    residual_l_values = possible_l_transfers(
        target_spin=residual_gs_spin,
        target_parity=residual_gs_parity,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity
    )

    for l in residual_l_values:
        c.add_coupling(
            channel_id_in=3,
            channel_id_out=4,
            l=l,
            s=0,
            coupling_one_way=True,
            ldfrm=3,
            icouex=True,
            beta=beta2_residual,
            icode=0,
        )

    # transfer from ground state in target to ground state in residual nucleus
    N, L, J = c.extract_orbital(GS_orbital)
    c.add_coupling(
        channel_id_in=1,
        channel_id_out=3,
        l=0,
        s=0,
        icode=2,
        ldfrm=0,
        coupling_one_way=True,
        icouex=False,
        beta=couplings[0],  # Coupling strength for GS to GS transfer
        cntrl=1.0,
        qcode=5.0,
        flmu=0.0,  # Default for two-nucleon transfer
        vzero=1.0, # Spectroscopic amplitude for two-nucleon transfer
        E=neutron_seperation_energy,
        mp=1.008,
        zp=0.0,
        mt=residual_mass,
        zt=residual_z,
        E_excited_state=0.0,  # Ground state, no excitation energy
        omp_lines="neutron",  # Use neutron potential
        N=N,
        L=L,
        J=J,
        S=1/2,  # Spin of the transferred particle
        VTRIAL=60.0,  # Trial potential scaling factor
        DAMP=0.0,  # Damping factor
    )

                # Add the coupling for the transfer from ground state in target to excited state in residual nucleus
    # determine the possible l values for the transfer
    direct_l_values = possible_l_transfers(
        target_spin=target_gs_spin,
        target_parity=target_gs_parity,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity
    )

    for l in direct_l_values:
        c.add_coupling(
            channel_id_in=1,
            channel_id_out=4,
            l=l,
            s=0,
            icode=2,
            ldfrm=0,
            coupling_one_way=True,
            icouex=False,
            beta=couplings[1],

            cntrl=1.0,
            qcode=5.0,
            vzero=1.0,

            E=neutron_seperation_energy,
            mp=1.008,
            zp=0.0,
            mt=residual_mass,
            zt=residual_z,
            E_excited_state=excited_state_energy/2, # Half if two-neutron transfer

            omp_lines="neutron",

            N=N,
            L=L,
            J=J,
            S=1/2,
            VTRIAL=60.0,
            FISW=0,
            DAMP=0.0,
        )


    for (i, l_transfer) in enumerate(target_excited_state_to_residual_excited_state_l_values):
        # Add the coupling for the transfer from excited state in target to excited state in residual nucleus
        c.add_coupling(
            channel_id_in=2,
            channel_id_out=4,
            l=l_transfer,
            s=0,
            icode=2,
            ldfrm=0,
            coupling_one_way=True,
            icouex=False,
            beta=couplings[i+2],  # Coupling strength
            cntrl=1.0,  # Read one set of orbital cards
            qcode=5.0,  # Two-nucleon transfer
            flmu=0.0,  # Default for two-nucleon transfer
            vzero=1.0,  # Spectroscopic amplitude for two-nucleon transfer
            E=neutron_seperation_energy,  # Neutron separation energy
            mp=1.008,  # Mass of the transferred particle (neutron)
            zp=0.0,  # Charge of the transferred particle (neutron)
            mt=residual_mass,  # Mass of the residual nucleus
            zt=residual_z,  # Charge of the residual nucleus
            E_excited_state=excited_state_energy/2,  # Half if two-neutron transfer
            omp_lines="neutron",  # Use neutron potential
            N=N,
            L=L,
            J=J,
            S=1/2,  # Spin of the transferred particle
            VTRIAL=60.0,  # Trial potential scaling factor
            FISW=0,  # No search on well depth
            DAMP=0.0,  # Damping factor
        )

    return c.build()

def two_step_config_2_input(
        excited_state_spin: float, 
        excited_state_parity: float, 
        excited_state_energy: float,
        couplings: list[float]
    ):

    '''
    Create a CHUCK3 input for a two-step configuration with specified couplings.

    Possible couplings:
    Target GS to Residual first 2+
    Residual First 2+ to Residual Excited State
    Target GS to First 2+ in Target
    First 2+ in Target to Residual Excited State (All possible l transfers)
    '''

    name = "_".join([f"{v:.0f}" for v in couplings])


    c = CHUCK3InputBuilder()
    c.set_header(differential_cross_section_log_scale=3, id=name) 

    # Target GS
    c.add_channel(
        id=1,
        target_spin=target_gs_spin,
        target_parity=target_gs_parity,
        energy=energy,
        projectile_mass=projectile_mass,
        projectile_charge=projectile_charge,
        target_mass=target_mass,
        target_z=target_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.85,
        projectile_spin=projectile_spin,
        optical_model_parameters=projectile_omp,
    )

    # Target excited state
    c.add_channel(
        id=2,
        target_spin=2,
        target_parity=1,
        energy=energy,
        q_value=0.0,
        excited_state_energy=0.130,
        projectile_mass=projectile_mass,
        projectile_charge=projectile_charge,
        target_mass=target_mass,
        target_z=target_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.85,
        projectile_spin=projectile_spin,
        optical_model_parameters=projectile_omp,
    )

    # First 2+ state in residual nucleus
    c.add_channel(
        id=3,
        target_spin=2,
        target_parity=1,
        energy=energy,
        q_value=Q_value,
        excited_state_energy=0.301,
        projectile_mass=ejectile_mass,
        projectile_charge=ejectile_charge,
        target_mass=residual_mass,
        target_z=residual_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.25,
        projectile_spin=ejectile_spin,
        optical_model_parameters=ejectile_omp,
    )

    # Residual nucleus excited state
    c.add_channel(
        id=4,
        target_spin=excited_state_spin,
        target_parity=excited_state_parity,
        energy=energy,
        q_value=Q_value,
        excited_state_energy=excited_state_energy,
        projectile_mass=ejectile_mass,
        projectile_charge=ejectile_charge,
        target_mass=residual_mass,
        target_z=residual_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.25,
        projectile_spin=ejectile_spin,
        optical_model_parameters=ejectile_omp,
    )


    # Coupling from target GS to first 2+ state in target
    # Collective coupling (qcode=0)
    c.add_coupling(
        channel_id_in=1,
        channel_id_out=2,
        l=2,
        s=0,
        coupling_one_way=True,
        ldfrm=3,
        icouex=True,
        beta=beta2_target,
        icode=0,
    )

    # Coupling from residual 2+ to excited state
    collective_residual_l_values = possible_l_transfers(
        target_spin=2,
        target_parity=1,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity
    )
    for l in collective_residual_l_values:
        c.add_coupling(
            channel_id_in=3,
            channel_id_out=4,
            l=l,
            s=0,
            coupling_one_way=True,
            ldfrm=3,
            icouex=True,
            beta=beta2_residual,
            icode=0,
        )

    # Direct: Coupling from target GS to residual excited state (2 neutron transfer)
    # First coupling value
    N, L, J = c.extract_orbital(GS_orbital)

    direct_l_values = possible_l_transfers(
        target_spin=target_gs_spin,
        target_parity=target_gs_parity,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity
    )
    for l in direct_l_values:
        c.add_coupling(
            channel_id_in=1,
            channel_id_out=4,
            l=l,
            s=0,
            icode=2,
            ldfrm=0,
            coupling_one_way=True,
            icouex=False,
            beta=couplings[0],

            cntrl=1.0,
            qcode=5.0,
            vzero=1.0,

            E=neutron_seperation_energy,
            mp=1.008,
            zp=0.0,
            mt=residual_mass,
            zt=residual_z,
            E_excited_state=excited_state_energy/2, # Half if two-neutron transfer

            omp_lines="neutron",

            N=N,
            L=L,
            J=J,
            S=1/2,
            VTRIAL=60.0,
            FISW=0,
            DAMP=0.0,
        )

    # Coupling from target GS to first 2+ state in residual nucleus (2 neutron transfer)
    # Second coupling value
    c.add_coupling(
        channel_id_in=1,
        channel_id_out=3,
        l=2,
        s=0,
        icode=2,
        ldfrm=0,
        coupling_one_way=True,
        icouex=False,
        beta=couplings[1],
        cntrl=1.0,
        qcode=5.0,
        flmu=0.0,
        vzero=1.0,
        E=neutron_seperation_energy,
        mp=1.008,
        zp=0.0,
        mt=residual_mass,
        zt=residual_z,
        E_excited_state=0.301/2, # Half if two-neutron transfer
        omp_lines="neutron",
        N=N,
        L=L,
        J=J,
        S=1/2,
        VTRIAL=60.0,
        FISW=0,
        DAMP=0.0,
    )

    # Coupling from target 2+ to residual excited state (2 neutron transfer)
    target_excited_state_to_residual_excited_state_l_values = possible_l_transfers(
        target_spin=2,
        target_parity=1,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity
    )
    for (i, l_transfer) in enumerate(target_excited_state_to_residual_excited_state_l_values):
        c.add_coupling(
            channel_id_in=2,
            channel_id_out=4,
            l=l_transfer,
            s=0,
            icode=2,
            ldfrm=0,
            coupling_one_way=True,
            icouex=False,
            beta=couplings[i+2],
            cntrl=1.0,
            qcode=5.0,
            flmu=0.0,
            vzero=1.0,
            E=neutron_seperation_energy,
            mp=1.008,
            zp=0.0,
            mt=residual_mass,
            zt=residual_z,
            E_excited_state=excited_state_energy/2,
            omp_lines="neutron",
            N=N,
            L=L,
            J=J,
            S=1/2,
            VTRIAL=60.0,
            FISW=0,
            DAMP=0.0,
        )

    return c.build()

def two_step_config_3_input(
        target_excited_state_spin: float,
        target_excited_state_parity: float,
        target_excited_state_energy: float,
        excited_state_spin_step: float,
        excited_state_parity_step: float,
        excited_state_energy_step: float,
        excited_state_spin: float, 
        excited_state_parity: float, 
        excited_state_energy: float,
        couplings: list[float]
    ):

    '''
    Create a CHUCK3 input for a two-step configuration with specified couplings.

    c0 = Coupling strength for the transfer from ground state in target to ground state in residual nucleus
    c1 = Coupling strength for the transfer from ground state in target to excited state step in residual nucleus
    c2 = Coupling strength for the transfer from ground state in target to excited state in residual nucleus
    c2, c3, ... = Coupling strengths for the transfer from excited state in target to excited state in residual nucleus
    '''

    name = "_".join([f"{v:.0f}" for v in couplings])

    target_excited_state_to_residual_excited_state_l_values = possible_l_transfers(
            target_spin=target_excited_state_spin,
            target_parity=target_excited_state_parity,
            excited_spin=excited_state_spin,
            excited_parity=excited_state_parity
        )
    
    target_l_values = possible_l_transfers(
            target_spin=target_spin,
            target_parity=target_parity,
            excited_spin=target_excited_state_spin,
            excited_parity=target_excited_state_parity
        )

    c = CHUCK3InputBuilder()
    c.set_header(differential_cross_section_log_scale=3, id=name) 

    # Target GS
    c.add_channel(
        id=1,
        target_spin=target_gs_spin,
        target_parity=target_gs_parity,
        energy=energy,
        projectile_mass=projectile_mass,
        projectile_charge=projectile_charge,
        target_mass=target_mass,
        target_z=target_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.85,
        projectile_spin=projectile_spin,
        optical_model_parameters=projectile_omp,
    )

    # Target excited state
    c.add_channel(
        id=2,
        target_spin=target_excited_state_spin,
        target_parity=target_excited_state_parity,
        energy=energy,
        q_value=0.0,
        excited_state_energy=target_excited_state_energy,
        projectile_mass=projectile_mass,
        projectile_charge=projectile_charge,
        target_mass=target_mass,
        target_z=target_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.85,
        projectile_spin=projectile_spin,
        optical_model_parameters=projectile_omp,
    )

    # Coupling from GS to excited state in target
    # Collective coupling (qcode=0)
    for l in target_l_values:
        c.add_coupling(
            channel_id_in=1,
            channel_id_out=2,
            l=l,
            s=0,
            coupling_one_way=True,
            ldfrm=3,
            icouex=True,
            beta=beta2_target,
            icode=0,
        )

    # Residual nucleus GS
    c.add_channel(
        id=3,
        target_spin=residual_gs_spin,
        target_parity=residual_gs_parity,
        energy=energy,
        q_value=Q_value,
        excited_state_energy=0.0,
        projectile_mass=ejectile_mass,
        projectile_charge=ejectile_charge,
        target_mass=residual_mass,
        target_z=residual_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.25,
        projectile_spin=ejectile_spin,
        optical_model_parameters=ejectile_omp,
    )

    # Residual nucleus excited state
    c.add_channel(
        id=4,
        target_spin=excited_state_spin,
        target_parity=excited_state_parity,
        energy=energy,
        q_value=Q_value,
        excited_state_energy=excited_state_energy,
        projectile_mass=ejectile_mass,
        projectile_charge=ejectile_charge,
        target_mass=residual_mass,
        target_z=residual_z,
        coulomb_charge_diffuseness=0.0,
        nonlocal_range_parameter=0.25,
        projectile_spin=ejectile_spin,
        optical_model_parameters=ejectile_omp,
    )

    # Coupling from residual GS to excited state
    # Collective coupling (qcode=0)
    residual_l_values = possible_l_transfers(
        target_spin=residual_gs_spin,
        target_parity=residual_gs_parity,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity
    )

    for l in residual_l_values:
        c.add_coupling(
            channel_id_in=3,
            channel_id_out=4,
            l=l,
            s=0,
            coupling_one_way=True,
            ldfrm=3,
            icouex=True,
            beta=beta2_residual,
            icode=0,
        )

    # transfer from ground state in target to ground state in residual nucleus
    N, L, J = c.extract_orbital(GS_orbital)
    c.add_coupling(
        channel_id_in=1,
        channel_id_out=3,
        l=0,
        s=0,
        icode=2,
        ldfrm=0,
        coupling_one_way=True,
        icouex=False,
        beta=couplings[0],  # Coupling strength for GS to GS transfer
        cntrl=1.0,
        qcode=5.0,
        flmu=0.0,  # Default for two-nucleon transfer
        vzero=1.0, # Spectroscopic amplitude for two-nucleon transfer
        E=neutron_seperation_energy,
        mp=1.008,
        zp=0.0,
        mt=residual_mass,
        zt=residual_z,
        E_excited_state=0.0,  # Ground state, no excitation energy
        omp_lines="neutron",  # Use neutron potential
        N=N,
        L=L,
        J=J,
        S=1/2,  # Spin of the transferred particle
        VTRIAL=60.0,  # Trial potential scaling factor
        DAMP=0.0,  # Damping factor
    )

                # Add the coupling for the transfer from ground state in target to excited state in residual nucleus
    # determine the possible l values for the transfer
    direct_l_values = possible_l_transfers(
        target_spin=target_gs_spin,
        target_parity=target_gs_parity,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity
    )

    for l in direct_l_values:
        c.add_coupling(
            channel_id_in=1,
            channel_id_out=4,
            l=l,
            s=0,
            icode=2,
            ldfrm=0,
            coupling_one_way=True,
            icouex=False,
            beta=couplings[1],

            cntrl=1.0,
            qcode=5.0,
            vzero=1.0,

            E=neutron_seperation_energy,
            mp=1.008,
            zp=0.0,
            mt=residual_mass,
            zt=residual_z,
            E_excited_state=excited_state_energy/2, # Half if two-neutron transfer

            omp_lines="neutron",

            N=N,
            L=L,
            J=J,
            S=1/2,
            VTRIAL=60.0,
            FISW=0,
            DAMP=0.0,
        )


    for (i, l_transfer) in enumerate(target_excited_state_to_residual_excited_state_l_values):
        # Add the coupling for the transfer from excited state in target to excited state in residual nucleus
        c.add_coupling(
            channel_id_in=2,
            channel_id_out=4,
            l=l_transfer,
            s=0,
            icode=2,
            ldfrm=0,
            coupling_one_way=True,
            icouex=False,
            beta=couplings[i+2],  # Coupling strength
            cntrl=1.0,  # Read one set of orbital cards
            qcode=5.0,  # Two-nucleon transfer
            flmu=0.0,  # Default for two-nucleon transfer
            vzero=1.0,  # Spectroscopic amplitude for two-nucleon transfer
            E=neutron_seperation_energy,  # Neutron separation energy
            mp=1.008,  # Mass of the transferred particle (neutron)
            zp=0.0,  # Charge of the transferred particle (neutron)
            mt=residual_mass,  # Mass of the residual nucleus
            zt=residual_z,  # Charge of the residual nucleus
            E_excited_state=excited_state_energy/2,  # Half if two-neutron transfer
            omp_lines="neutron",  # Use neutron potential
            N=N,
            L=L,
            J=J,
            S=1/2,  # Spin of the transferred particle
            VTRIAL=60.0,  # Trial potential scaling factor
            FISW=0,  # No search on well depth
            DAMP=0.0,  # Damping factor
        )

    return c.build()
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_objective, plot_evaluations


def two_step_config_1_bayesian_optimization(
    target_excited_state_spin,
    target_excited_state_parity,
    target_excited_state_energy,
    excited_state_spin,
    excited_state_parity,
    excited_state_energy,
    exp_angles,
    exp_cross_section,
    exp_cross_section_errors,
    input_file_path: str, 
    output_file_path: str,
    n_calls=100,
    n_jobs=6,
    coupling_num=4,
):
    print("Starting Bayesian optimization for CHUCK3 input generation...")

    progress_bar = tqdm(total=n_calls, desc="Optimizing", dynamic_ncols=True)
    best_info = {"chi2": np.inf, "rchi2": np.inf, "eta": None, "params": None}

    def progress_callback(res):
        best_so_far = best_info["rchi2"]
        current_chi2 = res.func_vals[-1]
        rchi2 = current_chi2
        progress_bar.set_postfix({
            "Current Δrχ²": f"{rchi2:.4f}",
            "Best Δrχ²": f"{abs(best_so_far - 1.0):.4f}",
        })
        progress_bar.update(1)

    # Automatically determine number of couplings
    target_to_exc_l = possible_l_transfers(
        target_spin=target_spin,
        target_parity=target_parity,
        excited_spin=target_excited_state_spin,
        excited_parity=target_excited_state_parity,
    )

    target_exc_to_final_l = possible_l_transfers(
        target_spin=target_excited_state_spin,
        target_parity=target_excited_state_parity,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity,
    )

    final_l = possible_l_transfers(
        target_spin=target_gs_spin,
        target_parity=target_gs_parity,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity,
    )

    n_couplings = 2 + len(target_exc_to_final_l)  # c0, c1, then one for each l-transfer in two-step

    space = [Real(-1560.0, 1560.0, name=f"c{i}") for i in range(n_couplings)]

    @use_named_args(space)
    def objective(**params):
        nonlocal best_info

        couplings = [params[f'c{i}'] for i in range(n_couplings)]
        input_str = two_step_config_1_input(
            target_excited_state_spin,
            target_excited_state_parity,
            target_excited_state_energy,
            excited_state_spin,
            excited_state_parity,
            excited_state_energy,
            couplings
        )
        with open("tmp.in", "w") as f:
            f.write(input_str)

        run_chuck3_input(chuck3_executable, "tmp.in", output_file="tmp.out")

        thetas_th, sigmas_th = parse_chuck3_output_for_optimization("tmp.out", coupling_num=coupling_num)

        if len(thetas_th) == 0 or len(sigmas_th) == 0:
            print("Warning: No CHUCK3 output found.")
            return np.inf
        
        # Interpolate theoretical sigmas to experimental angles
        interp = np.interp(exp_angles, thetas_th, sigmas_th, left=0, right=0)

        # Minimize chi2 by finding best scaling factor
        def chi2_scale(s):
            return np.sum(((s * interp - exp_cross_section) / exp_cross_section_errors) ** 2)

        result = minimize_scalar(chi2_scale, bounds=(1e-3, 1e10), method='bounded')
        if not result.success:
            print("Scaling failed.")
            return np.inf

        eta = result.x
        chi2 = chi2_scale(eta)
        dof = len(exp_angles) - 1
        rchi2 = chi2 / dof
        target_value = abs(rchi2 - 1.0)

        # Update best info
        if target_value < abs(best_info["rchi2"] - 1.0):
            best_info["chi2"] = chi2
            best_info["rchi2"] = rchi2
            best_info["eta"] = eta
            best_info["params"] = params.copy()

        return target_value
    
    result = gp_minimize(objective, 
                         space, 
                         n_calls=n_calls, 
                         random_state=0, 
                         verbose=False, 
                         n_jobs=n_jobs,
                         callback=[progress_callback],)

    print("\nOptimization Complete!")
    print(f"Best Chi²: {best_info['chi2']:.3f}")
    print(f"Best reduced Chi²: {best_info['rchi2']:.3f}")
    print(f"Best scaling factor (η): {best_info['eta']:.3f}")
    print("Best coupling strengths:")
    for key, val in best_info['params'].items():
        print(f"  {key}: {val:.2f}")

    # Save the best input file
    best_input_str = two_step_config_1_input(
        target_excited_state_spin,
        target_excited_state_parity,
        target_excited_state_energy,
        excited_state_spin,
        excited_state_parity,
        excited_state_energy,
        [best_info['params'][f'c{i}'] for i in range(n_couplings)]
    )

    with open(input_file_path, "w") as f:
        f.write(best_input_str)
        
    run_chuck3_input(chuck3_executable, input_file_path, output_file_path)

    # clean up temp files
    import os
    if os.path.exists("tmp.in"):
        os.remove("tmp.in")
    if os.path.exists("tmp.out"):
        os.remove("tmp.out")


    # Plot convergence: Chi² vs. iteration
    plot_convergence(result)
    # strip the stuff after the .
    name_base = os.path.splitext(output_file_path)[0]
    print(name_base)
    plt.savefig(f"{name_base}_bayesian_convergence.png", dpi=300)

    # Plot partial dependence of chi² on each parameter
    plot_objective(result)
    plt.savefig(f"{name_base}_bayesian_objective.png", dpi=300)

    # Plot pairwise parameter evaluations
    plot_evaluations(result)
    plt.savefig(f"{name_base}_bayesian_evaluations.png", dpi=300)

    return result

def two_step_config_2_bayesian_optimization(
    excited_state_spin,
    excited_state_parity,
    excited_state_energy,
    exp_angles,
    exp_cross_section,
    exp_cross_section_errors,
    input_file_path: str, 
    output_file_path: str,
    n_calls=100,
    n_jobs=6,
    coupling_num=4,
):
    print("Starting Bayesian optimization for CHUCK3 input generation...")

    progress_bar = tqdm(total=n_calls, desc="Optimizing", dynamic_ncols=True)
    best_info = {"chi2": np.inf, "rchi2": np.inf, "eta": None, "params": None}

    def progress_callback(res):
        best_so_far = best_info["rchi2"]
        current_chi2 = res.func_vals[-1]
        rchi2 = current_chi2
        progress_bar.set_postfix({
            "Current Δrχ²": f"{rchi2:.4f}",
            "Best Δrχ²": f"{abs(best_so_far - 1.0):.4f}",
        })
        progress_bar.update(1)


    target_exc_to_final_l = possible_l_transfers(
        target_spin=2,
        target_parity=1,
        excited_spin=excited_state_spin,
        excited_parity=excited_state_parity,
    )


    n_couplings = 2 + len(target_exc_to_final_l)  # c0, c1, then one for each l-transfer in two-step

    space = [Real(-1560.0, 1560.0, name=f"c{i}") for i in range(n_couplings)]

    @use_named_args(space)
    def objective(**params):
        nonlocal best_info

        couplings = [params[f'c{i}'] for i in range(n_couplings)]
        input_str = two_step_config_2_input(
            excited_state_spin,
            excited_state_parity,
            excited_state_energy,
            couplings
        )
        with open("tmp.in", "w") as f:
            f.write(input_str)

        run_chuck3_input(chuck3_executable, "tmp.in", output_file="tmp.out")

        thetas_th, sigmas_th = parse_chuck3_output_for_optimization("tmp.out", coupling_num=coupling_num)

        if len(thetas_th) == 0 or len(sigmas_th) == 0:
            print("Warning: No CHUCK3 output found.")
            return np.inf
        
        # Interpolate theoretical sigmas to experimental angles
        interp = np.interp(exp_angles, thetas_th, sigmas_th, left=0, right=0)

        # Minimize chi2 by finding best scaling factor
        def chi2_scale(s):
            return np.sum(((s * interp - exp_cross_section) / exp_cross_section_errors) ** 2)

        result = minimize_scalar(chi2_scale, bounds=(1e-3, 1e10), method='bounded')
        if not result.success:
            print("Scaling failed.")
            return np.inf

        eta = result.x
        chi2 = chi2_scale(eta)
        dof = len(exp_angles) - 1
        rchi2 = chi2 / dof
        target_value = abs(rchi2 - 1.0)

        # Update best info
        if target_value < abs(best_info["rchi2"] - 1.0):
            best_info["chi2"] = chi2
            best_info["rchi2"] = rchi2
            best_info["eta"] = eta
            best_info["params"] = params.copy()

        return target_value
    
    result = gp_minimize(objective, 
                         space, 
                         n_calls=n_calls, 
                         random_state=0, 
                         verbose=False, 
                         n_jobs=n_jobs,
                         callback=[progress_callback],)

    print("\nOptimization Complete!")
    print(f"Best Chi²: {best_info['chi2']:.3f}")
    print(f"Best reduced Chi²: {best_info['rchi2']:.3f}")
    print(f"Best scaling factor (η): {best_info['eta']:.3f}")
    print("Best coupling strengths:")
    for key, val in best_info['params'].items():
        print(f"  {key}: {val:.2f}")

    # Save the best input file
    best_input_str = two_step_config_2_input(
        excited_state_spin,
        excited_state_parity,
        excited_state_energy,
        [best_info['params'][f'c{i}'] for i in range(n_couplings)]
    )

    with open(input_file_path, "w") as f:
        f.write(best_input_str)
        
    run_chuck3_input(chuck3_executable, input_file_path, output_file_path)

    # clean up temp files
    import os
    if os.path.exists("tmp.in"):
        os.remove("tmp.in")
    if os.path.exists("tmp.out"):
        os.remove("tmp.out")


    # Plot convergence: Chi² vs. iteration
    plot_convergence(result)
    # strip the stuff after the .
    name_base = os.path.splitext(output_file_path)[0]
    print(name_base)
    plt.savefig(f"{name_base}_bayesian_convergence.png", dpi=300)

    # Plot partial dependence of chi² on each parameter
    plot_objective(result)
    plt.savefig(f"{name_base}_bayesian_objective.png", dpi=300)

    # Plot pairwise parameter evaluations
    plot_evaluations(result)
    plt.savefig(f"{name_base}_bayesian_evaluations.png", dpi=300)

    return result