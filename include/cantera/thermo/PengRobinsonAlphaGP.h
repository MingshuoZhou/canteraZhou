//! @file PengRobinsonAlphaGP.h

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#ifndef CT_PengRobinsonAlphaGP_H
#define CT_PengRobinsonAlphaGP_H

#include "MixtureFugacityTP.h"
#include "cantera/base/Array.h"

#include <vector>
#include <fstream>

#include "cantera/base/config.h"
#if CT_USE_SYSTEM_EIGEN
    #if CT_USE_SYSTEM_EIGEN_PREFIXED
    #include <eigen3/Eigen/Core>
    #include <eigen3/Eigen/Dense>
    #include <eigen3/Eigen/Sparse>
    #else
    #include <Eigen/Core>
    #include <Eigen/Dense>
    #include <Eigen/Sparse>
    #endif
#else
#include "cantera/ext/Eigen/Core"
#include "cantera/ext/Eigen/Dense"
#include "cantera/ext/Eigen/Sparse"
#endif

using namespace Eigen;

namespace Cantera
{

/**
 * Implementation of a Gaussian Process Model
 */
class GaussianProcess
{
public:
    explicit GaussianProcess(const std::string& datapath="", const std::string& species="");

    // added by Xingyu Su 2021/11/06
    void readCSV(const std::string filename, std::vector< std::vector<double> > &output, int &nrow, int &ncol);
    void readGPData(const std::string filename, MatrixXd &X, MatrixXd &y);
    void readGPPara(const std::string filename, VectorXd &BasisTheta, VectorXd &KernelGamma, double &KernelSigma, MatrixXd &HessianMatrix, double &yscale);
    void kernelFunc0(const MatrixXd &X1, const MatrixXd &X2,  MatrixXd &K) const;
	void kernelFunc(const MatrixXd &X1, const MatrixXd &X2,  MatrixXd &K) const;
    double predict(const MatrixXd &Xnew);
	double predictGP(const MatrixXd &Xnew);
	// added by Mingshuo
	void basisFunc(const MatrixXd &X1, MatrixXd &F) const;
	MatrixXd dfdtheta(const MatrixXd &X1) const;
	MatrixXd hmatrix(const MatrixXd &X1, const MatrixXd &X2) const;
protected:
    MatrixXd m_X;
    MatrixXd m_y;
    MatrixXd m_K;
    MatrixXd m_Ki;
    MatrixXd m_m;
	MatrixXd m_HessianMatrix;
    MatrixXd m_F; // f(X)

    VectorXd m_BasisTheta;
    VectorXd m_KernelGamma;
    double m_KernelSigmaF;
    double m_yscale;

    mutable MatrixXd m_Kx;
	mutable MatrixXd m_dfdtheta;
};

/**
 * Implementation of a multi-species Peng-Robinson equation of state
 *
 * @ingroup thermoprops
 */
class PengRobinsonAlphaGP : public MixtureFugacityTP
{
public:

    //! Construct and initialize a PengRobinsonAlphaGP object directly from an
    //! input file
    /*!
     * @param infile    Name of the input file containing the phase YAML data.
     *                  If blank, an empty phase will be created.
     * @param id        ID of the phase in the input file. If empty, the
     *                  first phase definition in the input file will be used.
     */
    explicit PengRobinsonAlphaGP(const std::string& infile="",
                          const std::string& id="");

    virtual std::string type() const {
        return "PengRobinsonAlphaGP";
    }

    //! @name Molar Thermodynamic properties
    //! @{

    virtual double cp_mole() const;
    virtual double cv_mole() const;

    //! @}
    //! @name Mechanical Properties
    //! @{

    //! Return the thermodynamic pressure (Pa).
    /*!
     * Since the mass density, temperature, and mass fractions are stored,
     * this method uses these values to implement the
     * mechanical equation of state \f$ P(T, \rho, Y_1, \dots, Y_K) \f$.
     *
     * \f[
     *    P = \frac{RT}{v-b_{mix}} - \frac{\left(\alpha a\right)_{mix}}{v^2 + 2b_{mix}v - b_{mix}^2}
     * \f]
     *
     * where:
     *
     * \f[
     *    \alpha = \left[ 1 + \kappa \left(1-T_r^{0.5}\right)\right]^2
     * \f]
     *
     *  and
     *
     * \f[
     *    \kappa = \left(0.37464 + 1.54226\omega - 0.26992\omega^2\right), \qquad \qquad \text{For } \omega <= 0.491 \\
     *    \kappa = \left(0.379642 + 1.487503\omega - 0.164423\omega^2 +  0.016667\omega^3 \right), \qquad \text{For } \omega > 0.491
     * \f]
     *
     * Coefficients \f$ a_{mix}, b_{mix} \f$ and \f$(a \alpha)_{mix}\f$ are calculated as
     *
     * \f[
     *    a_{mix} = \sum_i \sum_j X_i X_j a_{i, j} = \sum_i \sum_j X_i X_j \sqrt{a_i a_j}
     * \f]
     *
     * \f[
     *    b_{mix} = \sum_i X_i b_i
     * \f]
     *
     * \f[
     *   {a \alpha}_{mix} = \sum_i \sum_j X_i X_j {a \alpha}_{i, j} = \sum_i \sum_j X_i X_j \sqrt{a_i a_j} \sqrt{\alpha_i \alpha_j}
     * \f]
     */
    virtual double pressure() const;

    // @}

    //! Returns the standard concentration \f$ C^0_k \f$, which is used to
    //! normalize the generalized concentration.
    /*!
     * This is defined as the concentration by which the generalized
     * concentration is normalized to produce the activity.
     * The ideal gas mixture is considered as the standard or reference state here.
     * Since the activity for an ideal gas mixture is simply the mole fraction,
     * for an ideal gas,  \f$ C^0_k = P/\hat R T \f$.
     *
     * @param k Optional parameter indicating the species. The default is to
     *          assume this refers to species 0.
     * @return
     *   Returns the standard Concentration in units of m^3 / kmol.
     */
    virtual double standardConcentration(size_t k=0) const;

    //! Get the array of non-dimensional activity coefficients at the current
    //! solution temperature, pressure, and solution concentration.
    /*!
     * For all objects with the Mixture Fugacity approximation, we define the
     * standard state as an ideal gas at the current temperature and pressure of
     * the solution. The activities are based on this standard state.
     *
     * @param ac Output vector of activity coefficients. Length: m_kk.
     */
    virtual void getActivityCoefficients(double* ac) const;

    /// @name  Partial Molar Properties of the Solution
    //@{

    virtual void getChemPotentials(double* mu) const;
    virtual void getPartialMolarEnthalpies(double* hbar) const;
    virtual void getPartialMolarEntropies(double* sbar) const;
    virtual void getPartialMolarIntEnergies(double* ubar) const;
    //! Calculate species-specific molar specific heats
    /*!
     *  This function is currently not implemented for Peng-Robinson phase.
     */
    virtual void getPartialMolarCp(double* cpbar) const;
    virtual void getPartialMolarVolumes(double* vbar) const;

    //! Calculate species-specific critical temperature
    /*!
     *  The temperature dependent parameter in P-R EoS is calculated as
     *       \f[ T_{crit} = (0.0778 a)/(0.4572 b R) \f]
     *  Units: Kelvin
     *
     * @param a    species-specific coefficients used in P-R EoS
     * @param b    species-specific coefficients used in P-R EoS
     */
    virtual double speciesCritTemperature(double a, double b) const;
    virtual double speciesCritPressure(double a, double b) const;
    
    //! Set the temperature of the phase
    /*!
     * Currently this passes down to setState_TP(). It does not make sense to
     * calculate the standard state without first setting T and P.
     *
     * @param temp  Temperature (kelvin)
     */
    virtual void setTemperature(const doublereal temp);

    //! Set the internally stored pressure (Pa) at constant temperature and
    //! composition
    /*!
     * Currently this passes down to setState_TP().  It does not make sense to
     * calculate the standard state without first setting T and P.
     *
     *  @param p input Pressure (Pa)
     */
    virtual void setPressure(doublereal p);
    
    //@}
    //! @name Initialization Methods - For Internal use
    /*!
     * The following methods are used in the process of constructing
     * the phase and setting its parameters from a specification in an
     * input file. They are not normally used in application programs.
     * To see how they are used, see importPhase().
     */
    //@{

    virtual bool addSpecies(shared_ptr<Species> spec);
    virtual void initThermo();

    //! Retrieve \f$a\f$ and \f$b\f$ coefficients by looking up tabulated critical parameters
    /*!
    *  If `pureFluidParameters` are not provided for any species in the phase,
    *  consult the critical properties tabulated in `data/inputs/critProperties.xml`.
    *  If the species is found there, calculate pure fluid parameters \f$a_k\f$ and \f$b_k\f$ as:
    *  \f[ a_k = 0.4278 R^2 T_c^2 / P_c \f]
    *
    *  and:
    *  \f[ b_k = 0.08664 R T_c/ P_c \f]
    *
    *  @param iName    Name of the species
    */
    virtual vector_fp getCoeff(const std::string& iName);

    //! Set the pure fluid interaction parameters for a species
    /*!
     *  @param species   Name of the species
     *  @param a         \f$a\f$ parameter in the Peng-Robinson model [Pa-m^6/kmol^2]
     *  @param b         \f$a\f$ parameter in the Peng-Robinson model [m^3/kmol]
     *  @param w         acentric factor
     */
    void setSpeciesCoeffs(const std::string& species, double a, double b,
                          double w);

    //! Set values for the interaction parameter between two species
    /*!
     *  @param species_i   Name of one species
     *  @param species_j   Name of the other species
     *  @param a           constant term in the \f$a\f$ expression [Pa-m^6/kmol^2]
     *  @param alpha       dimensionless function of \f$T_r\f$ and \f$\omega\f$
     */
    void setBinaryCoeffs(const std::string& species_i,
                         const std::string& species_j, double a);

protected:
    // Special functions inherited from MixtureFugacityTP
    virtual double sresid() const;
    virtual double hresid() const;

    virtual double liquidVolEst(double TKelvin, double& pres) const;
    virtual double densityCalc(double T);

    virtual double densSpinodalLiquid() const;
    virtual double densSpinodalGas() const;
    virtual double dpdVCalc(double TKelvin, double molarVol, double& presCalc) const;

    // Special functions not inherited from MixtureFugacityTP

    //! Calculate temperature derivative \f$d(a \alpha)/dT\f$
    /*!
     *  These are stored internally.
     */
    double daAlpha_dT() const;

    //! Calculate second derivative \f$d^2(a \alpha)/dT^2\f$
    /*!
     *  These are stored internally.
     */
    double d2aAlpha_dT2() const;

public:
    //! Calculate \f$dp/dV\f$ and \f$dp/dT\f$ at the current conditions
    /*!
     *  These are stored internally.
     */
    void calculatePressureDerivatives() const;

    //! Update the \f$a\f$, \f$b\f$, and \f$\alpha\f$ parameters
    /*!
     *  The \f$a\f$ and the \f$b\f$ parameters depend on the mole fraction and the
     *  parameter $\f\alpha\f$ depends on the temperature. This function updates
     *  the internal numbers based on the state of the object.
     */
    virtual void updateMixingExpressions();

    //! Calculate the \f$a\f$, \f$b\f$, and \f$\alpha\f$ parameters given the temperature
    /*!
     * This function doesn't change the internal state of the object, so it is a
     * const function.  It does use the stored mole fractions in the object.
     *
     * @param aCalc (output)  Returns the a value
     * @param bCalc (output)  Returns the b value.
     * @param aAlpha (output) Returns the (a*alpha) value.
     */
    void calculateAB(double& aCalc, double& bCalc, double& aAlpha) const;

    void calcCriticalConditions(double& pc, double& tc, double& vc) const;

    //! Prepare variables and call the function to solve the cubic equation of state
    int solveCubic(double T, double pres, double a, double b, double aAlpha,
                   double Vroot[3]) const;
                   
    void updateAlpha(double T) const;
    
    void mixAlpha() const;

	inline double GPpredict(GaussianProcess GP, double T, int k) const;

protected:
    //! Value of \f$b\f$ in the equation of state
    /*!
     *  `m_b` is a function of the mole fractions and species-specific b values.
     */
    double m_b;

    //! Value of \f$a\f$ in the equation of state
    /*!
     *  `m_a` depends only on the mole fractions.
     */
    double m_a;

    //! Value of \f$\alpha\f$ in the equation of state
    /*!
     *  `m_aAlpha_mix` is a function of the temperature and the mole fractions.
     */
    double m_aAlpha_mix;

    // Vectors required to store species-specific a_coeff, b_coeff, alpha, kappa and other derivatives.
    // Length = m_kk
    vector_fp m_b_coeffs;
    vector_fp m_kappa;
    mutable vector_fp m_dalphadT;
    mutable vector_fp m_d2alphadT2;
    mutable vector_fp m_alpha;

    // Matrices for Binary coefficients a_{i,j} and {a*alpha}_{i.j} are saved in an Array form.
    // Length =  (m_kk, m_kk)
    Array2D m_a_coeffs;
    mutable Array2D m_aAlpha_binary;

    int m_NSolns;

    double m_Vroot[3];

    //! Temporary storage - length = m_kk.
    mutable vector_fp m_pp;

    // Partial molar volumes of the species
    mutable vector_fp m_partialMolarVolumes;

    //! The derivative of the pressure with respect to the volume
    /*!
     * Calculated at the current conditions. temperature and mole number kept
     * constant
     */
    mutable double m_dpdV;

    //! The derivative of the pressure with respect to the temperature
    /*!
     *  Calculated at the current conditions. Total volume and mole number kept
     *  constant
     */
    mutable double m_dpdT;

    //! Vector of derivatives of pressure with respect to mole number
    /*!
     *  Calculated at the current conditions. Total volume, temperature and
     *  other mole number kept constant
     */
    mutable vector_fp m_dpdni;

    mutable std::vector<GaussianProcess> m_alphaGP;
    mutable std::vector<GaussianProcess> m_cpGP;
    mutable std::vector<GaussianProcess> m_cpmoleGP;
    mutable std::vector<GaussianProcess> m_denGP;
    mutable std::vector<GaussianProcess> m_hresidGP;
    mutable std::vector<GaussianProcess> m_sresidGP;

private:
    //! Omega constant: a0 (= omega_a) used in Peng-Robinson equation of state
    /*!
     *  This value is calculated by solving P-R cubic equation at the critical point.
     */
    static const double omega_a;

    //! Omega constant: b0 (= omega_b) used in Peng-Robinson equation of state
    /*!
     *  This value is calculated by solving P-R cubic equation at the critical point.
     */
    static const double omega_b;

    //! Omega constant for the critical molar volume
    static const double omega_vc;
};

}

#endif