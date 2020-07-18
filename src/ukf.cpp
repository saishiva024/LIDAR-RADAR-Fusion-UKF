#include "ukf.h"
#include <iostream>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;

  aug_dim = 2 * n_aug_ + 1;

  lambda_ = 3 - n_aug_;

  std_a_ = 3.0;
  std_yawdd_ = 1.0;

  Xsig_pred_ = MatrixXd(n_x_, aug_dim);

  weights_ = VectorXd(aug_dim);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Make sure you switch between lidar and radar
     * measurements.
     */

    double delta_t;

    if (!is_initialized_)
    {
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];

            double radial_velocity = meas_package.raw_measurements_[2];

            double sin_phi = sin(phi);
            double cos_phi = cos(phi);

            double v = sqrt(
                        radial_velocity * sin_phi * radial_velocity * sin_phi +
                        radial_velocity * cos_phi * radial_velocity * cos_phi);

            P_ << 1, 0, 0, 0, 0, 
                  0, 1, 0, 0, 0, 
                  0, 0, 1, 0, 0, 
                  0, 0, 0, 1, 0, 
                  0, 0, 0, 0, 1;

            x_ << (rho * cos_phi), 
                  (rho * sin_phi), 
                  v, 
                  0, 
                  0;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            x_ << meas_package.raw_measurements_[0], 
                  meas_package.raw_measurements_[1], 
                  0, 
                  0, 
                  0;

            double std_laspx2 = std_laspx_ * std_laspx_;

            P_ << std_laspx2, 0, 0, 0, 0, 
                  0, std_laspx2, 0, 0, 0, 
                  0, 0, 5, 0, 0, 
                  0, 0, 0, 1, 0, 
                  0, 0, 0, 0, 1;
        }

        prev_time_stamp = meas_package.timestamp_;

        is_initialized_ = true;

        return;
    }

    weights_.fill(0.0);

    weights_(0) = lambda_ / (lambda_ + n_aug_);

    for (int i = 1; i < aug_dim; i++)
    {
        weights_(i) = 0.5 / (lambda_ + n_aug_);
    }

    delta_t = (meas_package.timestamp_ - prev_time_stamp) / 1000000.0;

    prev_time_stamp = meas_package.timestamp_;

    Prediction(delta_t);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
        UpdateLidar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    {
        UpdateRadar(meas_package);
    }
}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
    VectorXd x_aug = VectorXd(n_aug_);
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    MatrixXd Xsig_aug = MatrixXd(n_aug_, aug_dim);

    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0;
    x_aug(n_x_ + 1) = 0;

    P_aug.fill(0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

    MatrixXd sqrt_matrix = P_aug.llt().matrixL();

    Xsig_aug.fill(0);
    Xsig_aug.col(0) = x_aug;

    double shift = sqrt(lambda_ + n_aug_);

    for (int index = 0; index < n_aug_; index++)
    {
        Xsig_aug.col(index + 1) = x_aug + shift * sqrt_matrix.col(index);
        Xsig_aug.col(index + 1 + n_aug_) = x_aug - shift * sqrt_matrix.col(index);
    }

    for (int index = 0; index < aug_dim; index++)
    {
        double px = Xsig_aug(0, index);
        double py = Xsig_aug(1, index);
        double v  = Xsig_aug(2, index);
        double yaw = Xsig_aug(3, index);
        double yawd = Xsig_aug(4, index);
        double nu_a = Xsig_aug(5, index);
        double nu_yawdd = Xsig_aug(6, index);

        double p_px, p_py;

        if (fabs(yawd) > 0.001)
        {
            p_px = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            p_py = py + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else
        {
            p_px = px + v * delta_t * cos(yaw);
            p_py = py + v * delta_t * sin(yaw);
        }

        double p_v = v;
        double p_yaw = yaw + yawd * delta_t;
        double p_yawd = yawd;
        double multiplier_term = 0.5 * nu_a * delta_t * delta_t;

        p_px += multiplier_term * cos(yaw);
        p_py += multiplier_term * sin(yaw);
        p_v += nu_a * delta_t;

        p_yaw += 0.5 * nu_yawdd * delta_t * delta_t;
        p_yawd += nu_yawdd * delta_t;

        Xsig_pred_(0, index) = p_px;
        Xsig_pred_(1, index) = p_py;
        Xsig_pred_(2, index) = p_v;
        Xsig_pred_(3, index) = p_yaw;
        Xsig_pred_(4, index) = p_yawd;
    }

    x_.fill(0.0);

    for (int index = 0; index < aug_dim; index++)
    {
        x_ += weights_(index) * Xsig_pred_.col(index);
    }

    P_.fill(0.0);

    for (int index = 0; index < aug_dim; index++)
    {
        VectorXd x_diff = Xsig_pred_.col(index) - x_;

        while(x_diff(3) > M_PI)
        {
            x_diff(3) -= 2.0 * M_PI;
        }
        while(x_diff(3) < -M_PI)
        {
            x_diff(3) += 2.0 * M_PI;
        }

        P_ += weights_(index) * x_diff * x_diff.transpose();
    }
}

double UKF::CalculateNIS(VectorXd p_Z, VectorXd Z, MatrixXd covar)
{
    VectorXd diff = Z - p_Z;
    
    return diff.transpose() * covar.inverse() * diff;
}


void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    VectorXd Z = meas_package.raw_measurements_;

    MatrixXd H = MatrixXd(2, n_x_);

    H << 1, 0, 0, 0, 0, 
         0, 1, 0, 0, 0;
    
    VectorXd p_Z = H * x_;

    MatrixXd R = MatrixXd(2, 2);

    R << std_laspx_ * std_laspx_, 0, 
         0,                       std_laspy_ * std_laspy_;

    VectorXd Y = Z - p_Z;
    
    MatrixXd S = H * P_ * H.transpose() + R;
    
    MatrixXd K = P_ * H.transpose() * S.inverse();

    x_ += K * Y;

    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

    P_ = (I - (K * H)) * P_;

    std::cout  << "LIDAR NIS - " << CalculateNIS(p_Z, Z, S) << std::endl;
}

void UKF::Normalize(VectorXd &list, int index, bool negativePi=false)
{
    while(list(index) > M_PI)
    {
        list(index) -= 2.0 * M_PI;
    }

    if(negativePi)
    {
        while(list(index) < -M_PI)
        {
            list(index) += 2.0 * M_PI;
        }
    }
    else
    {
        while(list(index) < M_PI)
        {
            list(index) += 2.0 * M_PI;
        }
    }
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
    int n_Z = 3;

    VectorXd Z = meas_package.raw_measurements_;

    MatrixXd Zsig = MatrixXd(n_Z, aug_dim);

    VectorXd p_Z = VectorXd(n_Z);

    MatrixXd R = MatrixXd(n_Z, n_Z);

    MatrixXd S = MatrixXd(n_Z, n_Z);

    MatrixXd cross_correlation = MatrixXd(n_x_, n_Z);

    for (int i = 0; i < aug_dim; i++)
    {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double vx = v * cos(yaw);
        double vy = v * sin(yaw);

        double dist = sqrt(pow(px, 2) + pow(py, 2));

        Zsig(0, i) = dist;
        Zsig(1, i) = atan2(py, px);
        Zsig(2, i) = (px * vx + py * vy) / dist;
    }

    p_Z.fill(0);

    for (int i = 0; i < aug_dim; i++)
    {
        p_Z += weights_(i) * Zsig.col(i);
    }

    S.fill(0);

    for (int i = 0; i < aug_dim; i++)
    {
        VectorXd z_diff = Zsig.col(i) - p_Z;

        Normalize(z_diff, 1);

        S += weights_(i) * z_diff * z_diff.transpose();
    }

    double std_radr2 = std_radr_ * std_radr_;

    R << std_radr2, 0,         0, 
         0,         std_radr2, 0, 
         0,         0,         std_radr2;

    S = S + R;

    cross_correlation.fill(0);

    for (int i = 0; i < aug_dim; i++)
    {
        VectorXd z_diff = Zsig.col(i) - p_Z;

        Normalize(z_diff, 1);

        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Normalize(x_diff, 1, true);

        cross_correlation += weights_(i) * x_diff * z_diff.transpose();
    }

    MatrixXd K = cross_correlation * S.inverse();

    VectorXd z_diff = Z - p_Z;

    Normalize(z_diff, 1);

    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    std::cout << "RADAR NIS - " << CalculateNIS(p_Z, Z, S) << std::endl;
}