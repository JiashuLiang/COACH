#include <cmath>
#include <algorithm>

namespace _
{
size_t get_nbuff_eval_mgga_exc(const size_t nbsfA, const size_t nbsfB)
{
    const size_t nDF = 1 + N_D1_META, nDFV = 10;
    size_t nstep1 = get_nbuff_mgga_DFV(nbsfA, nbsfB);
    size_t nstep2 = 2 * nDF;
    return nDFV + std::max(nstep1, nstep2);
}

size_t nele_series = 96;
size_t nseries = 18;
size_t nseries_x = 2*nseries;
size_t nseries_ss = 2*nseries;
size_t nseries_os = nseries;

// Generate Linear series (1, x, x^2, ...) given x and order vec.n_elem, output to vec
void Linear(arma::vec &vec, double x) {
    size_t n = vec.n_elem;
    vec(0) = 1.0;
    for(int i = 1; i < n; i++) {
        vec(i) = vec(i-1) * x;
    }
}
// Generate Chebyshev series given x and order vec.n_elem >=1, output to vec
void Chebyshev(arma::vec &vec, double x) {
    size_t n = vec.n_elem;
    vec(0) = 1.0;
    vec(1) = x;
    for(int i = 2; i < n; i++) {
        vec(i) = 2.0 * x * vec(i-1) - vec(i-2);
    }
}
// Generate Chebyshev series given x and order vec.n_elem >=1, output to vec
void Legendre(arma::vec &vec, double x) {
    size_t n = vec.n_elem;
    vec(0) = 1.0;
    vec(1) = x;
    for(int i = 2; i < n; i++) {
        double i_d = double(i);
        vec(i) = ( (2.0*i_d-1.0)*x*vec(i-1) - (i_d-1.0)*vec(i-2)) /i_d;
    }
}

// generate various expansion using the above functions and the variable u, w, beta_f
void Expansion(arma::mat &basis_sub, double u, double w, double beta_f){

    // To store the expansion of u
    arma::vec u_series_linear(8);
    arma::vec u_series_legendre(8);
    arma::vec u_series_chebyshev(8);
    // To store the expansion of mGGA-related variable w
    arma::vec w_series_linear(12);
    arma::vec w_series_legendre(12);
    arma::vec w_series_chebyshev(12);
    // To store the expansion of mGGA-related variable beta_f
    arma::vec beta_series_linear(12);
    arma::vec beta_series_legendre(12);
    arma::vec beta_series_chebyshev(12);

    
    //Fill the series
    Linear(u_series_linear, u);
    Legendre(u_series_legendre, u);
    Chebyshev(u_series_chebyshev, u);
    Linear(w_series_linear, w);
    Legendre(w_series_legendre, w);
    Chebyshev(w_series_chebyshev, w);
    Linear(beta_series_linear, beta_f);
    Legendre(beta_series_legendre, beta_f);
    Chebyshev(beta_series_chebyshev, beta_f);
    
    // Fill the expansion basis matrix
    basis_sub.col(0) = arma::kron(w_series_linear, u_series_linear);
    basis_sub.col(1) = arma::kron(w_series_legendre, u_series_linear);
    basis_sub.col(2) = arma::kron(w_series_chebyshev, u_series_linear);
    basis_sub.col(3) = arma::kron(w_series_linear, u_series_legendre);
    basis_sub.col(4) = arma::kron(w_series_legendre, u_series_legendre);
    basis_sub.col(5) = arma::kron(w_series_chebyshev, u_series_legendre);
    basis_sub.col(6) = arma::kron(w_series_linear, u_series_chebyshev);
    basis_sub.col(7) = arma::kron(w_series_legendre, u_series_chebyshev);
    basis_sub.col(8) = arma::kron(w_series_chebyshev, u_series_chebyshev);

    basis_sub.col(9) = arma::kron(beta_series_linear, u_series_linear);
    basis_sub.col(10) = arma::kron(beta_series_legendre, u_series_linear);
    basis_sub.col(11) = arma::kron(beta_series_chebyshev, u_series_linear);
    basis_sub.col(12) = arma::kron(beta_series_linear, u_series_legendre);
    basis_sub.col(13) = arma::kron(beta_series_legendre, u_series_legendre);
    basis_sub.col(14) = arma::kron(beta_series_chebyshev, u_series_legendre);
    basis_sub.col(15) = arma::kron(beta_series_linear, u_series_chebyshev);
    basis_sub.col(16) = arma::kron(beta_series_legendre, u_series_chebyshev);
    basis_sub.col(17) = arma::kron(beta_series_chebyshev, u_series_chebyshev);

}

    
void eval_exchange_channel(const size_t ngrid, const arma::vec &weights,
    const arma::vec &Rho, const arma::mat &Rho1, const arma::vec &Tau, arma::mat &exchange_terms) {

    exchange_terms.zeros();
    arma::mat exchange_mGGA(exchange_terms.colptr(0), nele_series, nseries_x, false, true);
    arma::mat exchange_wmGGA(exchange_terms.colptr(nseries_x), nele_series, nseries_x, false, true);
    arma::mat basis_tmp(nele_series, nseries, arma::fill::zeros);
    
    double Tol = 1e-14;
    const double Pi = M_PI;
    const double E = M_E;
    double cLDA = -(3.0/2.0)*(std::pow(3.0/(4.0*Pi),1.0/3.0));
    // double cGGA = 1.0/(4.0*std::pow(6.0*(std::pow(Pi,2)),2.0/3.0));
    double tau_UEG_coeffecient = (3.0/5.0)*(std::pow(6.0*Pi*Pi,2.0/3.0));
    double omega = 0.3;
    double gamma_x = 0.004;
    double sqrtPi = std::sqrt(Pi);
    double kF_coeffecient = std::pow(6.0*Pi*Pi,1.0/3.0);


//  double caa = 0.85;
//  double cab = 0.259;
//  double cac = 1.007;
    
    for(size_t i = 0; i < ngrid; i++) {
        double RA = std::max(Rho[i],double(0.0));
        double GA = std::pow(Rho1(i,0),2)+std::pow(Rho1(i,1),2)+std::pow(Rho1(i,2),2);
        double TA = std::max(Tau[i],double(0.0));
        if((RA > Tol) && (TA > Tol)) {
        
            double Rho4o3 = std::pow(RA,4.0/3.0);
            double Rho1o3 = std::pow(RA,1.0/3.0);
            double exUEG = cLDA*Rho4o3;
            
            double a = omega/(kF_coeffecient*Rho1o3);
            double a3 = std::pow(a,3);
            double am2 = std::pow(a,-2);
            double ainv = std::pow(a,-1);
            double Fn = 1.0-(2.0/3.0)*a *(a3 - 3.0*a + (2.0*a- a3)/std::pow(E, am2)+ 2.0*sqrtPi * std::erf(ainv));

            double s2 = GA/std::pow(RA,8.0/3.0);
            double u = (gamma_x*s2)/(1.0+gamma_x*s2);
            double taua_UEG = tau_UEG_coeffecient * std::pow(RA,5.0/3.0);
            double ts = taua_UEG / TA;
            double w = (-1.0+ts)/(1.0+ts);
            double taua_W= GA/RA/4.0;
            double beta_f = 2.0 * (TA- taua_W)/(TA + taua_UEG) - 1.0; // (TA- 2*taua_W -taua_UEG)/(TA + taua_UEG);
            Expansion(basis_tmp, u, w, beta_f);

            double wexUEG =  exUEG * weights[i];
            double wFnexUEG = wexUEG * Fn;
            exchange_mGGA.cols(0,nseries-1) += wexUEG * basis_tmp;
            exchange_wmGGA.cols(0,nseries-1) += wFnexUEG * basis_tmp;

            // Add a multiplier to satisfy the nonuniform scaling
            // The coefficient is converted from SCANs's: (2(6\pi^2)^{1/3})^(1/2) * 4.9479 = 13.815
            double nonuniform_scaling_factor = 1.0 - std::exp(-13.815 / std::pow(s2, 1.0/4.0));
            exchange_mGGA.cols(nseries,2*nseries-1) += (nonuniform_scaling_factor * wexUEG) * basis_tmp;
            exchange_wmGGA.cols(nseries,2*nseries-1) += (nonuniform_scaling_factor * wFnexUEG) * basis_tmp;

        }
    }  
}

inline double G_func(double rs, double A, double alpha_1, double beta_1, double beta_2, double beta_3, double beta_4){
    return -2.0 * A * (1.0 + alpha_1 * rs) * 
            std::log(1.0 + 1.0/(2.0*A*( (beta_1 + beta_3 * rs) * std::sqrt(rs) + (beta_2 + beta_4 * rs)* rs)));
}

// Evaluate the SCAN correlation energy correction to LDA when beta_f = 0
double H_func(double rs, double s2, double zeta, double e_LDA){
    double phi = (std::pow(1.0-zeta,2.0/3.0) +std::pow(1.0+zeta,2.0/3.0))/2.0;
    double gamma = (1.0-std::log(2.0))/(M_PI*M_PI);
    double gammaphi3 = gamma * std::pow(phi,3);

    double beta_rs = 0.066725*(1.0+0.1*rs)/(1.0+0.1778*rs);
    double w1 = std::exp(-e_LDA/ gammaphi3) - 1.0;

    double A =beta_rs / w1 /gamma;
    double t2 = s2/(16*std::pow(4.0,1.0/3.0) * rs*phi*phi);

    double G_At2 = std::pow(1.0 + 4*A * t2, -0.25);
    double H1 = gammaphi3 * std::log(1.0 + w1* (1.0 - G_At2));

    return H1; 

}

void eval_correlation_channels(const size_t ngrid, const arma::vec &weights,
    const arma::vec &RhoA, const arma::vec &RhoB,
    const arma::mat &RhoA1, const arma::mat &RhoB1,
    const arma::vec &TauA, const arma::vec &TauB,
    arma::mat &correlation_terms){

    assert(utils::check_arma(RhoA, ngrid));
    assert(utils::check_arma(RhoB, ngrid));
    assert(utils::check_arma(RhoA1, ngrid, 3));
    assert(utils::check_arma(RhoB1, ngrid, 3));
    assert(utils::check_arma(TauA, ngrid));
    assert(utils::check_arma(TauB, ngrid));
    assert(utils::check_arma(correlation_terms, nele_series, 2*(nseries_ss + nseries_os)));

    arma::mat corr_ss(correlation_terms.colptr(0), nele_series, nseries_ss, false, true);
    arma::mat corr_os(correlation_terms.colptr(nseries_ss), nele_series, nseries_os, false, true);
    arma::mat corr_ss_SCAN(correlation_terms.colptr(nseries_ss + nseries_os), nele_series, nseries_ss, false, true);
    arma::mat corr_os_SCAN(correlation_terms.colptr(2*nseries_ss + nseries_os), nele_series, nseries_os, false, true);
    arma::mat basis_tmp(nele_series, nseries, arma::fill::zeros);
   
    double Tol = 1e-14;
    const double Pi = M_PI;
    const double E = M_E;
    double rs_coeffecient = std::pow(0.75/Pi,1.0/3.0);
    double tau_UEG_coeffecient = (3.0/5.0)*(std::pow(6.0*Pi*Pi,2.0/3.0));
    double total_tau_UEG_coeffecient = (3.0/5.0)*(std::pow(3.0*Pi*Pi,2.0/3.0));
    double gamma_ss = 0.2;
    double gamma_os = 0.006;

    double fppz = 4.0/(9.0*(std::pow(2.0,1.0/3.0)-1.0));

    
    corr_os.zeros();
    for(size_t i = 0; i < ngrid; i++) {
        double RA = std::max(RhoA[i],double(0.0));
        double RB = std::max(RhoB[i],double(0.0));
        double GA = std::pow(RhoA1(i,0),2)+std::pow(RhoA1(i,1),2)+std::pow(RhoA1(i,2),2);
        double GB = std::pow(RhoB1(i,0),2)+std::pow(RhoB1(i,1),2)+std::pow(RhoB1(i,2),2);
        double TA = std::max(TauA[i],double(0.0));
        double TB = std::max(TauB[i],double(0.0));

        // ss alpha spin
        double eps_c_PW92_a = 0.0, eps_c_SCAN_a = 0.0;
        double s2a = 0.0, tsa = 0.0;
        if((RA > Tol) && (TA > Tol)) {
            double rs = rs_coeffecient * std::pow(RA,-1.0/3.0);
            eps_c_PW92_a = G_func(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517);
            s2a = GA/std::pow(RA,8.0/3.0);
            eps_c_SCAN_a = eps_c_PW92_a + H_func(rs, s2a, 1.0, eps_c_PW92_a);

            double u = (gamma_ss*s2a)/(1.0+gamma_ss*s2a);
            double taua_UEG = tau_UEG_coeffecient * std::pow(RA,5.0/3.0);
            tsa = taua_UEG / TA;
            double w = (-1.0+tsa)/(1.0+tsa);
            double taua_W= GA/RA/4.0;
            double beta = (TA- taua_W)/(TA + taua_UEG);
            double beta_f = 2.0 * beta - 1.0;
            Expansion(basis_tmp, u, w, beta_f);
            
            double we_c_PW92_a = eps_c_PW92_a * RA * weights[i];
            double we_c_SCAN_a = eps_c_SCAN_a * RA * weights[i];
            corr_ss.cols(0,nseries-1) += we_c_PW92_a * basis_tmp;
            corr_ss_SCAN.cols(0,nseries-1) += we_c_SCAN_a * basis_tmp;

            // Add a multiplier to satisfy the one-electron self-interaction correction using beta
            corr_ss.cols(nseries,2*nseries-1) += (2.0 * beta * we_c_PW92_a) * basis_tmp;
            corr_ss_SCAN.cols(nseries,2*nseries-1) += (2.0 * beta * we_c_SCAN_a) * basis_tmp;
            
        }

        // ss beta spin
        double eps_c_PW92_b = 0.0, eps_c_SCAN_b = 0.0;
        double s2b = 0.0, tsb = 0.0;
        if((RB > Tol) && (TB > Tol)) {
            double rs = rs_coeffecient * std::pow(RB,-1.0/3.0);
            eps_c_PW92_b = G_func(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517);
            s2b = GB/std::pow(RB,8.0/3.0);
            eps_c_SCAN_b = eps_c_PW92_b + H_func(rs, s2b, -1.0, eps_c_PW92_b);


            double u = (gamma_ss*s2b)/(1.0+gamma_ss*s2b);
            double taub_UEG = tau_UEG_coeffecient * std::pow(RB,5.0/3.0);
            tsb = taub_UEG / TB;
            double w = (-1.0+tsb)/(1.0+tsb);
            double taub_W= GB/RB/4.0;
            double beta = (TB- taub_W)/(TB + taub_UEG);
            double beta_f = 2.0 * beta - 1.0;
            Expansion(basis_tmp, u, w, beta_f);
            
            double we_c_PW92_b = eps_c_PW92_b * RB * weights[i];
            double we_c_SCAN_b = eps_c_SCAN_b * RB * weights[i];
            corr_ss.cols(0,nseries-1) += we_c_PW92_b * basis_tmp;
            corr_ss_SCAN.cols(0,nseries-1) += we_c_SCAN_b * basis_tmp;

            // Add a multiplier to satisfy the one-electron self-interaction correction using beta
            corr_ss.cols(nseries,2*nseries-1) += (2.0 * beta * we_c_PW92_b) * basis_tmp;
            corr_ss_SCAN.cols(nseries,2*nseries-1) += (2.0 * beta * we_c_SCAN_b) * basis_tmp;
        }

        // os
        if((RA > Tol) && (RB > Tol) && (TA > Tol) && (TB > Tol)) {
            double R = RA + RB;
            double G = std::pow(RhoA1(i,0)+RhoB1(i,0),2)
                + std::pow(RhoA1(i,1)+RhoB1(i,1),2)
                + std::pow(RhoA1(i,2)+RhoB1(i,2),2);
            double T = TA + TB;
            
            double rs = rs_coeffecient * std::pow(R, -1.0/3.0);
            double Alpla_c = G_func(rs, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671);
            double GPW_0 = G_func(rs, 0.0310907, 0.2137, 7.5957, 3.5876, 1.6382, 0.49294);
            double GPW_1 = G_func(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517);

            double zeta = (RA-RB)/R;
            if(std::abs(zeta) < Tol) 
                zeta = 0.0;
            double fpol = (-2.0 + std::pow(1.0-zeta,4.0/3.0) +std::pow(1.0+zeta,4.0/3.0)) /(-2.0 + 2.0 * std::pow(2.0,1.0/3.0));
            double zeta4 = std::pow(zeta,4);
            double eps_c_PW92 = GPW_0 +(fpol * Alpla_c* (zeta4 -1.0))/fppz + fpol* (GPW_1 - GPW_0) *zeta4;

            double e_c_PW92_os = eps_c_PW92*R - eps_c_PW92_a * RA - eps_c_PW92_b * RB;
            double we_c_PW92_os = e_c_PW92_os * weights[i];
            
            double s2 = G / std::pow(R,8.0/3.0);
            double eps_c_SCAN = eps_c_PW92 + H_func(rs, s2, zeta, eps_c_PW92);
            double e_c_SCAN_os = eps_c_SCAN*R - eps_c_SCAN_a * RA - eps_c_SCAN_b * RB;
            double we_c_SCAN_os = e_c_SCAN_os * weights[i];
            
            double s2_average = (s2a+ s2b) / 2.0;
            double u = (gamma_os*s2_average)/(1.0+gamma_os*s2_average);
            double ts = (tsa + tsb) / 2.0;
            double w = (-1.0+ts)/(1.0+ts);

            // I prefer to use total density to calculate beta_f for os
            double tau_UEG = total_tau_UEG_coeffecient * std::pow(R,5.0/3.0);
            double tau_W= G/R/4.0;
            double beta_f = 2.0 * (T- tau_W)/(T + tau_UEG) - 1.0;
            Expansion(basis_tmp, u, w, beta_f);

            corr_os += we_c_PW92_os * basis_tmp;
            corr_os_SCAN += we_c_SCAN_os * basis_tmp;
        }
    }
}


void eval_integrated_DV(const size_t ngrid,
    const arma::vec &grid_weights,
    const arma::vec &Rhoa, const arma::vec &Rhob,
    const arma::mat &Rhoa1, const arma::mat &Rhob1,
    const arma::vec &Taua, const arma::vec &Taub,
    arma::mat &integratedDV_batch) {

    assert(utils::check_arma(Rhoa, ngrid));
    assert(utils::check_arma(Rhob, ngrid));
    assert(utils::check_arma(Rhoa1, ngrid, 3));
    assert(utils::check_arma(Rhob1, ngrid, 3));
    assert(utils::check_arma(Taua, ngrid));
    assert(utils::check_arma(Taub, ngrid));

    
    integratedDV_batch.zeros();
    
    arma::mat exchange_a(integratedDV_batch.colptr(0), nele_series, 2*nseries_x, false, true);
    arma::mat exchange_b(integratedDV_batch.colptr(2*nseries_x), nele_series, 2*nseries_x, false, true);
    eval_exchange_channel(ngrid, grid_weights, Rhoa, Rhoa1, Taua, exchange_a);
    eval_exchange_channel(ngrid, grid_weights, Rhob, Rhob1, Taub, exchange_b);
    exchange_a += exchange_b;

    // Reuse exchange_b memory
    exchange_b.zeros();
    arma::mat corr_terms(integratedDV_batch.colptr(2*nseries_x), nele_series, 2*(nseries_ss + nseries_os), false, true);
    eval_correlation_channels(ngrid, grid_weights, Rhoa, Rhob, Rhoa1, Rhob1, Taua, Taub, corr_terms);
    
}
    
}
