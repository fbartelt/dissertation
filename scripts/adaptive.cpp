#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
// #include <functional>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>

using namespace Eigen;

// Global variables
std::vector<MatrixXd> nearest_points;
std::vector<VectorXd> xiT_hist, xiN_hist;
double GAIN_N1 = 1.0, GAIN_N2 = 30.0, GAIN_T1 = 0.1*0, GAIN_T2 = 15.0;
double epsilon = 1e-3, ds = 1e-3;

Eigen::Matrix3d skew(const Eigen::Vector3d& q) {
  Eigen::Matrix3d skewMatrix;
  skewMatrix << 0, -q(2), q(1), q(2), 0, -q(0), -q(1), q(0), 0;
  return skewMatrix;
}

void printProgressBar(int current, int imax) {
    // Calculate percentage
    int percent = static_cast<int>(100.0 * current / imax);

    // Calculate the number of "=" to show in the progress bar
    int barWidth = 50;  // Width of the progress bar in characters
    int pos = barWidth * current / imax;

    // Create the progress bar
    std::string progressBar = "[" + std::string(pos, '=') + std::string(barWidth - pos, ' ') + "]";

    // Print the progress bar with the percentage
    std::cout << "\r" << progressBar << " " << percent << "%";
    std::cout.flush();  // Ensure the output is immediately printed
}

void saveToCSV(const std::string& filename,
               const std::vector<MatrixXd>& nearest_points,
               const std::vector<MatrixXd>& points,
               const std::vector<VectorXd>& xiT_hist,
               const std::vector<VectorXd>& xiN_hist,
               const std::vector<VectorXd>& dq_hist,
               const std::vector<double>& norm_s_hist) {
    // Open file stream
    std::ofstream file(filename);

    // Check if file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Iterate over nearest_points vector to write matrices and vectors in the correct format
    for (size_t i = 0; i < nearest_points.size(); ++i) {
        const MatrixXd& mat = nearest_points[i];
        const MatrixXd& point = points[i];
        const VectorXd& xiT = xiT_hist[i];
        const VectorXd& xiN = xiN_hist[i];
        const VectorXd& dq = dq_hist[i];

        // Write the roation matrix
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                file << mat(row, col);
                file << ",";
            }
        }
        // Write the position vector
        for (int row = 3; row < 6; ++row) {
            file << mat(row, 6) << ",";
        }

        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                file << point(row, col);
                file << ",";
            }
        }
        // Write the position vector
        for (int row = 3; row < 6; ++row) {
            file << point(row, 6) << "," ;
        }

        // Write the 3D vector (xiT)
        for (int j = 0; j < xiT.size(); ++j) {
            file << xiT(j) << "," ;
        }

        // Write the 3D vector (xiN)
        for (int j = 0; j < xiN.size(); ++j) {
            file << xiN(j) << "," ;
        }

        // Write the 6D vector (dq)
        for (int j = 0; j < dq.size(); ++j) {
            file << dq(j) << "," ;
        }

        // Write the norm of s
        file << norm_s_hist[i];

        // Add a newline after each iteration
        file << std::endl;
    }

    // Close the file stream
    file.close();
}

// Maps a 3x1 vector to the custom iota matrix
MatrixXd iota(const Eigen::Vector3d& v) {
  // Create the iota matrix (3x6) as a dynamic-sized matrix
  MatrixXd LMatrix(3, 6);
  LMatrix << v(0), v(1), v(2), 0, 0, 0,
             0, v(0), 0, v(1), v(2), 0,
             0, 0, v(0), 0, v(1), v(2);
  return LMatrix;
  }

// Converts a nx1 vector into a homogeneous transformation matrix (Translation
// group T(n))
Eigen::MatrixXd Rn2Tn(const Eigen::VectorXd& vector) {
  int n = vector.size();
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n + 1, n + 1);
  H.block(0, n, n, 1) = vector;  // Set the translation vector
  return H;
}

// Converts a position vector and orientation into an ISE(3) block-diagonal
// matrix
Eigen::MatrixXd to_ise3(const Eigen::Vector3d& p, const Eigen::Matrix3d& R) {
  Eigen::MatrixXd p_tilde = Rn2Tn(p);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(7, 7);
  // Fill the block diagonal matrix
  H.block(0, 0, 3, 3) = R;        // Top-left block (Rotation matrix)
  H.block(3, 3, 4, 4) = p_tilde;  // Bottom-right block (Rn2Tn result)

  return H;
}

// Computes the inverse of an ISE(3) block-diagonal matrix
Eigen::MatrixXd ise3_inv(const Eigen::MatrixXd& H) {
  Eigen::Matrix3d R = H.block(0, 0, 3, 3);  // Extract rotation
  Eigen::Vector3d p = H.block(3, 6, 3, 1);  // Extract translation vector

  Eigen::MatrixXd p_tilde = Rn2Tn(-p);
  Eigen::MatrixXd H_inv = Eigen::MatrixXd::Zero(7, 7);

  // Construct the block-diagonal inverse
  MatrixXd R_tranpose = R.transpose().eval();
  H_inv.block(0, 0, 3, 3) = R_tranpose;  // Transpose of rotation matrix
  H_inv.block(3, 3, 4, 4) = p_tilde;        // Inverted translation component

  return H_inv;
}

MatrixXd hd(double s, double c1, double h0) {
    double theta = 2 * M_PI * s;
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    // Identity matrix for hds
    MatrixXd hds = MatrixXd::Identity(7, 7);

    // Compute p
    Vector3d p;
    p << c1 * (sin_theta + 2 * std::sin(2 * theta)),
         c1 * (cos_theta - 2 * std::cos(2 * theta)),
         h0 + c1 * (-std::sin(3 * theta));

    // Compute rotation matrices
    Matrix3d Rz, Rx, R;
    Rz << cos_theta, -sin_theta, 0,
          sin_theta,  cos_theta, 0,
          0,          0,         1;
    // double theta_x = 2 * theta;
    Rx << 1, 0,          0,
          0, std::cos(2*theta), -std::sin(2*theta),
          0, std::sin(2*theta),  std::cos(2*theta);

    R = Rz * Rx;
    // ----------------- MAJOR CHANGE -----------------
    // R = Matrix3d::Identity();
    // TODO: CHANGED

    // Assign to hds
    hds = to_ise3(p, R);

    return hds;
}

MatrixXd hd_dot(double s, double c1, double h0) {
    double theta = 2 * M_PI * s;
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double s2t = std::sin(2 * theta);
    double c2t = std::cos(2 * theta);

    // Identity matrix for hds
    MatrixXd hds = MatrixXd::Zero(7, 7);
    Vector3d p_dot;
    p_dot << c1 * (cos_theta + 4*cos(2*theta)),
             c1 * (-sin_theta + 4*sin(2*theta)),
             c1 * (-3*cos(3*theta));

    Matrix3d R_dot;
    R_dot << -sin_theta, 2*sin_theta*s2t - cos_theta*c2t, 2*sin_theta*c2t + s2t*cos_theta,
             cos_theta,  -2*sin_theta*c2t -2*s2t*cos_theta, sin_theta*s2t - 2*c2t*cos_theta,
             0,          2*c2t,                        -2*s2t;

    hds = to_ise3(p_dot, R_dot);
    return hds;
}

std::vector<MatrixXd> precompute_curve(
    std::function<MatrixXd(double, double, double)> fun,
    int n_points, double c1, double h0) {

    std::vector<MatrixXd> precomputed;
    std::vector<double> s_values(n_points);

    // Generate s values linearly from 0 to 1
    for (int i = 0; i < n_points; ++i) {
        s_values[i] = static_cast<double>(i) / (n_points - 1);
    }

    // Compute the curve for each s
    for (double s : s_values) {
        precomputed.push_back(fun(s, c1, h0));
    }

    return precomputed;
}

double EEdistance(const MatrixXd& V, const MatrixXd& W) {
  MatrixXd Z = ise3_inv(V) * W;
  Matrix3d Q = Z.block(0, 0, 3, 3);
  Vector3d u = Z.block(3, 6, 3, 1);
  double costheta = (Q.trace() - 1) / 2;
  MatrixX3d Q_transpose = Q.transpose().eval();
  double sintheta = 1/(2 * std::sqrt(2)) * (Q - Q_transpose).norm();
  double theta = std::atan2(sintheta, costheta);
  return std::sqrt(2 * theta * theta + pow(u.norm(), 2));
}

std::tuple<double, int> ECdistance(const std::vector<MatrixXd>& curve, const Vector3d& p, const Matrix3d& R) {
  double distance = 1e6;
  int ind_min = 0;
  for (int i = 0; i < curve.size(); ++i) {
    MatrixXd Hds = curve[i];
    double d = EEdistance(to_ise3(p, R), Hds);
    if (d < distance) {
      distance = d;
      ind_min = i;
    }
  }
  return {distance, ind_min};
}

MatrixXd S(const Eigen::VectorXd& xi) {
    Eigen::MatrixXd lie_algebra = Eigen::MatrixXd::Zero(7, 7); // Initialize the Lie algebra element

    // The last three elements of xi are assigned to the upper right 3x3 block
    lie_algebra(0, 1) = -xi(5);
    lie_algebra(0, 2) = xi(4);
    lie_algebra(1, 2) = -xi(3);

    // Since the orientation-related portion of the Lie algebra is skew-symmetric:
    lie_algebra = lie_algebra - lie_algebra.transpose().eval();

    // The first three elements of xi are assigned to the last column
    lie_algebra.block<3, 1>(3, 6) = xi.head(3);

    return lie_algebra;
}

VectorXd S_inv(const Eigen::MatrixXd& lie_algebra) {
    // Ensure lie_algebra is a 7x7 matrix
    Eigen::VectorXd xi = Eigen::VectorXd::Zero(6); // Initialize the twist vector

    // The last three elements of xi are assigned from the upper right 3x3 block
    xi(3) = -lie_algebra(1, 2);
    xi(4) = lie_algebra(0, 2);
    xi(5) = -lie_algebra(0, 1);

    // The first three elements of xi are assigned from the last column
    xi.head(3) = lie_algebra.block<3, 1>(3, 6);

    return xi;
}

double kn(double distance){
  return GAIN_N1 * std::tanh(GAIN_N2 * distance);
}

double kt(double distance){
  return GAIN_T1 * (1.0 - std::tanh(GAIN_T2 * distance));
}

VectorXd twist_d(const VectorXd& p, const MatrixXd& R, const std::vector<MatrixXd>& curve, bool store_points) {
  //Computes ECdistance between the curve and the current pose
  auto [distance, ind_min] = ECdistance(curve, p, R);
  MatrixXd Hd_star = curve[ind_min];
  MatrixXd H = to_ise3(p, R);

  // Compute normal component using L operator
  VectorXd L_ = VectorXd::Zero(6);
  MatrixXd I = MatrixXd::Identity(6, 6);
  for (int i = 0; i < 6; ++i) {
    MatrixXd deltaV = (S(I.col(i)) * epsilon).exp() * H;
    L_(i) = (EEdistance(deltaV, Hd_star) - distance) / epsilon;
  }
  VectorXd xi_N = -L_; // L_ is already computed as column vector

  // Compute tangent component
  MatrixXd Hd_next;
  if (ind_min == curve.size() - 1) {
    Hd_next = curve[0];
  }
  else {
    Hd_next = curve[ind_min + 1];
  }
  MatrixXd dHds = (Hd_next - Hd_star) / ds;
  VectorXd xi_T = S_inv(dHds * ise3_inv(Hd_star));

  // Compute the twist
  xi_N = kn(distance) * xi_N;
  xi_T = kt(distance) * xi_T;
  VectorXd psi = xi_N + xi_T;

  if (store_points) {
    // Store the points
    nearest_points.push_back(Hd_star);
    xiT_hist.push_back(xi_T);
    xiN_hist.push_back(xi_N);
    std::cout << "Distance: " << distance << std::endl;
    std::cout << "xi_N norm: " << xi_N.norm() << std::endl;
    std::cout << "xi_T norm: " << xi_T.norm() << std::endl;
  }

  return psi;
}

class AdaptiveController {
 public:
  // Class properties to replace global variables
  MatrixXd P_o, P_r, Kd;
  VectorXd a_i;
  Vector3d r_p;
  Matrix3d I_p;
  float m;
  int N;
  double tol;
  std::vector<MatrixXd> aprox_hist;
  std::vector<VectorXd> input_hist;
  std::vector<Vector3d> r_i;
  std::vector<std::vector<VectorXd>> taui_hist;

  AdaptiveController(const MatrixXd& P_o_init, const MatrixXd& P_r_init,
                     const MatrixXd& Kd_init, const VectorXd& a_i_init,
                     std::vector<Vector3d> r_i_init, const Matrix3d& I_p_init,
                     const float m_init, const int N_init,
                     const Vector3d& r_p_init)
      : P_o(P_o_init),
        P_r(P_r_init),
        Kd(Kd_init),
        a_i(a_i_init),
        r_i(r_i_init),
        I_p(I_p_init),
        m(m_init),
        N(N_init),
        r_p(r_p_init) {}

  std::tuple<VectorXd, std::vector<VectorXd>, std::vector<VectorXd>>
  adaptive_dyn(const MatrixXd& x, const MatrixXd& x_d, const MatrixXd& R,
               const MatrixXd& R_d, const MatrixXd& dq,
               const std::vector<VectorXd>& a_hat,
               const std::vector<Vector3d>& r_hat, 
               const std::vector<MatrixXd>& curve, double t,
               const float dt = 0.001, VectorXd psi = VectorXd(),
               bool store_tau = false) {
    Vector3d dx = dq.block<3, 1>(0, 0);
    Vector3d w = dq.block<3, 1>(3, 0);

    double norm_vel = psi.norm();

    if (psi.size() == 0) {
      psi = twist_d(x_d, R_d, curve, false); // TODO: CHANGED -- x, R -> x_d, R_d
    }

    // Reference signals
    Vector3d w_r = psi.segment<3>(3);
    Vector3d v_r = psi.segment<3>(0);
    VectorXd psi_next =
        twist_d(x + v_r * dt, (Matrix3d(skew(w_r) * dt).exp() * R), curve, false); // TODO: CHANGED -- x, R -> x_d, R_d; dx, w -> v_r, w_r
    VectorXd psi_back =
        twist_d(x - v_r * dt, (Matrix3d(skew(-w_r) * dt).exp() * R), curve, false); // TODO: CHANGED 
    VectorXd psi_dot = (psi_next - psi_back) / (2 * dt * norm_vel); // TODO: CHANGED psi -> psi_back, dt -> 2dt
    aprox_hist.push_back(psi_dot);
    // Vector3d dx_t = dx - psi_dot.segment<3>(0);
    // Vector3d x_t = x - x_d;
    VectorXd s = dq - psi;
    Vector3d al_r = psi_dot.segment<3>(3);
    Vector3d a_r = psi_dot.segment<3>(0);
    // VectorXd ddq_r = psi_dot;
    // VectorXd dq_r = psi;

    // Compute regressors
    MatrixXd Y_l(3, 10);
    Y_l.block<3, 1>(0, 0) = a_r;
    Y_l.block<3, 3>(0, 1) = -skew(al_r) * R - skew(w) * skew(w_r) * R;
    Y_l.block<3, 6>(0, 4) = MatrixXd::Zero(3, 6);

    MatrixXd R_transpose = R.transpose().eval();

    MatrixXd Y_r(3, 10);
    Y_r.block<3, 1>(0, 0) = MatrixXd::Zero(3, 1);  // First column is zero
    Y_r.block<3, 3>(0, 1) =
        skew(a_r) * R + skew(w) * skew(v_r) * R - skew(w_r) * skew(dx) * R;
    Y_r.block<3, 6>(0, 4) = R * iota(R_transpose * al_r) +
                            skew(w) * R * iota(R_transpose * w_r);

    MatrixXd Y_o(6, 10);
    // Concatenate Y_l and Y_r into Y_o
    Y_o.block<3, 10>(0, 0) = Y_l;
    Y_o.block<3, 10>(3, 0) = Y_r;

    // True dynamics matrices
    MatrixXd H(6, 6), C(6, 6), H_dot(6, 6);
    // Compute blocks for H
    H.block<3, 3>(0, 0) = m * Matrix3d::Identity();  // Top-left block
    H.block<3, 3>(0, 3) = m * skew(R * r_p);         // Top-right block
    H.block<3, 3>(3, 0) = -m * skew(R * r_p);        // Bottom-left block
    H.block<3, 3>(3, 3) = R * I_p * R.transpose().eval();   // Bottom-right block

    // Compute blocks for C
    C.block<3, 3>(0, 0) = Matrix3d::Zero();              // Top-left block
    C.block<3, 3>(0, 3) = m * skew(w) * skew(R * r_p);   // Top-right block
    C.block<3, 3>(3, 0) = -m * skew(w) * skew(R * r_p);  // Bottom-left block
    C.block<3, 3>(3, 3) =
        skew(w) * R * I_p * R_transpose  // Bottom-right block
        - m * skew(skew(R * r_p) * dx);
    // Matrix3d off_diag = m * skew(w) * R * skew(r_p) * R.transposeInPlace() - m * R *
    // skew(r_p) * R.transposeInPlace() * skew(w); H_dot << Matrix3d::Zero(), off_diag,
    //         -off_diag, skew(w) * R * I_p * R.transposeInPlace() - R * I_p *
    //         R.transposeInPlace() * skew(w);

    // Adaptive control law
    VectorXd input = VectorXd::Zero(6);
    std::vector<VectorXd> F(N, VectorXd::Zero(6));
    std::vector<VectorXd> kth_taui;

    for (int i = 0; i < N; ++i) {
      // MatrixXd G(6, 6);
      // G << Matrix3d::Identity(), Matrix3d::Zero(),
      //     skew(R * r_i[i]), Matrix3d::Identity();

      // MatrixXd G_h(6, 6);
      // G_h << Matrix3d::Identity(), Matrix3d::Zero(),
      //       skew(R * r_hat[i]), Matrix3d::Identity();

      MatrixXd G_h_inv(6, 6);
      G_h_inv.block<3, 3>(0, 0) = Matrix3d::Identity();
      G_h_inv.block<3, 3>(0, 3) = Matrix3d::Zero();
      G_h_inv.block<3, 3>(3, 0) = -skew(R * r_hat[i]);
      G_h_inv.block<3, 3>(3, 3) = Matrix3d::Identity();

      F[i] = Y_o * a_hat[i] - Kd * s;
      VectorXd tau = G_h_inv * F[i];
      kth_taui.push_back(tau);
      input += tau;
    }

    if (store_tau) {
      taui_hist.push_back(kth_taui);
      input_hist.push_back(input);
    }

    VectorXd ddq = H.inverse() * (input - C * dq);

    // Adaptation laws
    // MatrixXd Y_g(N, MatrixXd(6, 3)), dr(N, VectorXd(3));
    // MatrixXd a_t(N, VectorXd(10)), r_t(N, VectorXd(3));
    // MatrixXd da(N, VectorXd(10)), g_o(N, MatrixXd(10, 10)), g_r(N,
    // MatrixXd(3, 3));
    std::vector<MatrixXd> Y_g(N, MatrixXd(6, 3)), g_o(N, MatrixXd(10, 10)),
        g_r(N, MatrixXd(3, 3));
    std::vector<VectorXd> dr(N, VectorXd(3)), a_t(N, VectorXd(10)),
        r_t(N, VectorXd(3)), da(N, VectorXd(10));

    for (int i = 0; i < N; ++i) {
      Y_g[i].block<3, 3>(0, 0) = Matrix3d::Zero();
      Y_g[i].block<3, 3>(3, 0) = skew(F[i].head(3)) * R;
      g_o[i] = P_o.inverse();
      g_r[i] = P_r.inverse();

      MatrixXd Y_o_transpose = Y_o.transpose().eval();
      MatrixXd Y_gi_tranpose = Y_g[i].transpose().eval();
      da[i] = -g_o[i] * Y_o_transpose * s;
      dr[i] = -g_r[i] * Y_gi_tranpose * s;
      a_t[i] = a_hat[i] - a_i;
      r_t[i] = r_hat[i] - r_i[i];
    }

    return {ddq, da, dr};
  }
};

int main() {
  // Constants
  double rho = 8050.0;  // Density of steel in kg/m^3
  double r = 0.25;      // Radius of the cylinder in meters
  double h = 1.0;       // Height of the cylinder in meters
  double m = rho * M_PI * std::pow(r, 2) * h;  // Mass of the cylinder in kg

  // Random number generator
  std::mt19937 rng(42);
  std::normal_distribution<> normal_dist(0.0, 1.0);

  // Fixed sample
  Vector3d r_p(0.0, 0.0, h / 2.0);  // Measurement point

  int N = 6;  // Number of agents

  // Initial positions of the agents
  std::vector<Vector3d> r_i(N, Vector3d::Zero());
  r_i[0] << 0, 0, h / 2;
  r_i[1] << 0, 0, -h / 2;
  r_i[2] << r, 0, 0;
  r_i[3] << -r, 0, 0;
  r_i[4] << 0, r, 0;
  r_i[5] << 0, -r, 0;

  // Inertia tensor
  Matrix3d I_cm = (1.0 / 12.0) * Matrix3d::Identity();
  I_cm(0, 0) *= m * (3 * std::pow(r, 2) + std::pow(h, 2));
  I_cm(1, 1) *= m * (3 * std::pow(r, 2) + std::pow(h, 2));
  I_cm(2, 2) *= 6 * m * std::pow(r, 2);

  Matrix3d I_p = I_cm - m * skew(r_p) * skew(r_p);

  // std::vector<MatrixXd> a_hat(N, MatrixXd::Zero(10, 1));
  std::vector<VectorXd> a_hat(N, VectorXd::Zero(10));
  for (auto& a : a_hat) {
    for (int i = 0; i < 10; ++i) {
      a(i, 0) = normal_dist(rng);
    }
  }

  // True ai
  VectorXd a_I(6);
  a_I << I_p(0, 0), I_p(0, 1), I_p(0, 2), I_p(1, 1), I_p(1, 2), I_p(2, 2);
  VectorXd a(10);
  a << m, m * r_p, a_I;
  VectorXd a_i = (1.0 / N) * a;

  // Initial r_hat
  std::vector<Vector3d> r_hat(N, Vector3d::Zero());
  for (auto& r : r_hat) {
    r = 2.0 * Vector3d(normal_dist(rng), normal_dist(rng), normal_dist(rng));
  }

  // Kd
  float k = 10e-1; // old 1e-1, 1 works for normal only
  MatrixXd Kd = MatrixXd::Identity(6, 6);
  Kd.diagonal().head(3).setConstant(k * (5e3 / N)); //5e4, 5e3 works for normal only
  Kd.diagonal().tail(3).setConstant(k * (5e3 / N));

  // P_o
  VectorXd abs_a_i = a_i.cwiseAbs() + 1e-2 * VectorXd::Ones(a_i.size());
  MatrixXd P_o = 1e1 * abs_a_i.cwiseInverse().asDiagonal().inverse(); //3e1

  // P_r
  MatrixXd P_r = 3e3 * Matrix3d::Identity(); // 3e3

  // tol
  double tol = 1e-5;

  // Adaptive controller
  AdaptiveController controller(P_o, P_r, Kd, a_i, r_i, I_p, m, N, r_p);


  // Precompute the curve
  double c1 = 0.7, h0 = 0.4; // 0.7, 0.4
  int n_points = 5000;
  std::vector<MatrixXd> curve = precompute_curve(hd, n_points, c1, h0);
  // std::vector<MatrixXd> curve_derivative = precompute_curve(hd_dot, n_points, c1, h0);

  // Initial conditions
  // TOOD: CHANGE HERE
  // r_hat = r_i;
  // for (auto& a: a_hat) {
  //   a = a_i;
  // }
  Vector3d x;
  x << -0.1, 0, 0.2;
  Matrix3d R = Matrix3d::Identity();
  // MatrixXd Htest = curve[0];
  // x = Htest.block(3, 6, 3, 1);
  // Matrix3d R = Htest.block(0, 0, 3, 3);
  Vector3d x_d = x;
  Matrix3d R_d = R;
  MatrixXd H_ref = to_ise3(x_d, R_d);
  MatrixXd H_real = to_ise3(x, R);
  VectorXd dq = VectorXd::Zero(6);
  Vector3d w = dq.tail(3); // angular velocity
  Vector3d dx = dq.head(3); // linear velocity
  VectorXd s = VectorXd::Zero(6);

  // Store points
  std::vector<MatrixXd> H_hist;
  std::vector<VectorXd> dq_hist;
  std::vector<double> norm_s_hist;

  // Iterate system
  double T = 20;
  double dt = 1e-3;
  int imax = T / dt;
  double deadband = 0.01 / 10; // TODO: CHANGED -- 0.01 * 5 (WORKS)

  for (int i = 0; i < imax; ++i) {
    printProgressBar(i, imax);
    // Compute the twist
    VectorXd psi = twist_d(x, R, curve, true); // TODO: CHANGED -- x, R -> x_d, R_d

    // Compute the next pose
    s = dq - psi;
    std::cout << "s norm: " << s.norm() << std::endl;

    // First step of Heun -- Euler method
    auto [ddq, da, dr] = controller.adaptive_dyn(x, x_d, R, R_d, dq, a_hat, r_hat, curve, i * dt, dt, psi, true);

    std::vector<VectorXd> a_int(N, VectorXd::Zero(10));
    std::vector<Vector3d> r_int(N, Vector3d::Zero());

    for (int j=0; j < N; ++j) {
      if (s.norm() > deadband){
        a_int[j] = a_hat[j] + da[j] * dt;
        r_int[j] = r_hat[j] + dr[j] * dt;
        // a_hat[j] += da[j] * dt;
        // r_hat[j] += dr[j] * dt;
      }
      else {
        da[j] = VectorXd::Zero(10);
        dr[j] = Vector3d::Zero();
        a_int[j] = a_hat[j];
        r_int[j] = r_hat[j];
      }
      
    }

    // Second Step of Heuns Method
    MatrixXd H_ref_int = (dt * S(psi)).exp() * H_real; // TODO: CHANGED --- H_ref -> H_real
    MatrixXd H_real_int = (dt * S(dq)).exp() * H_real;
    Matrix3d R_d_int = H_ref_int.block(0, 0, 3, 3);
    Vector3d x_d_int = H_ref_int.block(3, 6, 3, 1);
    Matrix3d R_int = H_real_int.block(0, 0, 3, 3);
    Vector3d x_int = H_real_int.block(3, 6, 3, 1);
    VectorXd dq_int = dq + ddq * dt;
    VectorXd psi_int = twist_d(x_int, R_int, curve, false);

    auto [ddq_int, da_int, dr_int] = controller.adaptive_dyn(x_int, x_d_int, R_int, R_d_int, dq_int, a_int, r_int, curve, (i + 1) * dt, dt, psi_int, false);
    for (int j=0; j < N; ++j) {
      if (s.norm() > deadband){
        a_hat[j] += 0.5 * (da[j] + da_int[j]) * dt;
        r_hat[j] += 0.5 * (dr[j] + dr_int[j]) * dt;
      }
      else {
        da_int[j] = VectorXd::Zero(10);
        dr_int[j] = Vector3d::Zero();
      }
    }

    // Update pose 
    H_ref = (0.5 * dt * S(psi + psi_int)).exp() * H_real; // TODO: CHANGED --- H_ref -> H_real
    H_real = (0.5 * dt * S(dq + dq_int)).exp() * H_real;
    dq += 0.5 * dt * (ddq + ddq_int);


    // Print the norm of the error between r_hat and r_i, and a_hat and a_i
    VectorXd error_a = VectorXd::Zero(N);
    VectorXd error_r = VectorXd::Zero(N);
    VectorXd norm_da = VectorXd::Zero(N);
    VectorXd norm_dr = VectorXd::Zero(N);
    for (int j = 0; j < N; ++j) {
      error_a(j) = (a_hat[j] - a_i).norm();
      error_r(j) = (r_hat[j] - r_i[j]).norm();
      norm_da(j) = da[j].norm();
      norm_dr(j) = dr[j].norm();
    }
    std::cout << "Error a norm: " << error_a.norm()  << ".  Error r norm: " << error_r.norm() << ".  Norm da: " << norm_da.norm() << ".  Norm dr: " << norm_dr.norm() << std::endl;

    // H_ref = (dt * S(psi)).exp() * H_ref;
    // H_real = (dt * S(dq)).exp() * H_real;
    // dq += ddq * dt;
    H_ref = H_real; // TODO: MAJOR CHANGE -- H_ref -> H_real (mirror an MPC behavior)
    R_d = H_ref.block(0, 0, 3, 3);
    x_d = H_ref.block(3, 6, 3, 1);
    R = H_real.block(0, 0, 3, 3);
    x = H_real.block(3, 6, 3, 1);
    dx = dq.head(3);
    w = dq.tail(3);
    H_hist.push_back(H_real);
    dq_hist.push_back(dq);
    norm_s_hist.push_back(s.norm());
  }

  saveToCSV("cpp_adaptive.csv", nearest_points, H_hist, xiT_hist, xiN_hist, dq_hist, norm_s_hist);

  return 0;
}
