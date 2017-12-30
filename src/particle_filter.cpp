/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 500;
    weights = vector<double>(num_particles, 1.0);

    particles = {};
    for (int i = 0; i < num_particles; ++i) {
        Particle p = {};
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        p.associations = {};
        p.sense_x = {};
        p.sense_y = {};
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    const double EPS = 0.0001;

    for(size_t i = 0; i < particles.size(); i++){
        Particle p = particles[i];
        const  double d_theta = yaw_rate * delta_t;

        double x_next;
        double y_next;

        if(yaw_rate < EPS){
            x_next = p.x + velocity * cos(p.theta) * delta_t ;
            y_next = p.y + velocity * sin(p.theta) * delta_t ;
        }else{
            x_next = p.x + (velocity / yaw_rate) * (sin(p.theta + d_theta) - sin(p.theta));
            y_next = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + d_theta));
        }
        const double theta_next = p.theta + d_theta;

        // add noise
        normal_distribution<double> dist_x(x_next, std_pos[0]);
        normal_distribution<double> dist_y(y_next, std_pos[1]);
        normal_distribution<double> dist_theta(theta_next, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

// gaussian function
inline double gaussian(double x, double mu, double sig){
    return (1.0/sqrt(2*M_PI))*exp(-(x-mu)*(x-mu)/(2.0*sig*sig));
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    const double std_x = std_landmark[0];
    const double std_y = std_landmark[1];

    vector<LandmarkObs> landmarks = {};
    for(size_t i = 0; i < map_landmarks.landmark_list.size(); i++){
        LandmarkObs landmark = {};
        landmark.id = map_landmarks.landmark_list[i].id_i;
        landmark.x = map_landmarks.landmark_list[i].x_f;
        landmark.y = map_landmarks.landmark_list[i].y_f;
        landmarks.push_back(landmark);
    }

    double total_weights = 0.0;

    // loop for particles
    for(size_t i = 0; i < particles.size(); i++){
        Particle p  = particles[i];

        size_t obs_size = observations.size();
        vector<int> associates = vector<int>(obs_size);
        vector<double> sense_x = vector<double>(obs_size);
        vector<double> sense_y = vector<double>(obs_size);

        // loop for observations in the current particle
        double weight_next = 1.0;
        for(size_t j = 0; j < obs_size; j++) {
            LandmarkObs obs = observations[j];
            LandmarkObs tobs = {};
            tobs.id = obs.id;
            tobs.x = cos(p.theta) * obs.x - sin(p.theta) * obs.y + p.x;
            tobs.y = sin(p.theta) * obs.x + cos(p.theta) * obs.y + p.y;

            // find closest landmark for this transformed observation.
            // check distances of all landmarks which defined the map
            double min_distance = -1;
            LandmarkObs closest_lm;
            for(size_t k = 0; k < landmarks.size(); k++){
                LandmarkObs lm = landmarks[k];
                double d = dist(lm.x, lm.y, tobs.x, tobs.y);
                //cout << "particle: " << p.id << ", id:" << obs.id << ", min_d: " << min_distance << endl;
                if(min_distance < 0){
                    // enter here only once.
                    min_distance = d;
                    closest_lm = lm;
                }else if(d < min_distance){
                    min_distance = d;
                    closest_lm = lm;
                }
            }
            // calculate weight of j-th observation
            const double weight_j = gaussian(tobs.x, closest_lm.x, std_x) * gaussian(tobs.y, closest_lm.y, std_y);

            // next weight is given by products of all observation
            weight_next *= weight_j;

            // store observation result of the particle.
            associates[j] = closest_lm.id;
            sense_x[j] = tobs.x;
            sense_y[j] = tobs.y;
        }
        SetAssociations(p, associates, sense_x, sense_y);

        // update particle weight
        particles[i].weight = weight_next;
        total_weights += weight_next;

        //cout << "particle_id: " << p.id << ", weight: " << p.weight << endl;
    }

    //set normalized weights;
    for(size_t i = 0; i < particles.size(); i++) {
        particles[i].weight = particles[i].weight / total_weights;;
    }

}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;

    for(size_t i = 0; i < particles.size(); i++){
        weights[i] = particles[i].weight;
    }
    std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());

    vector<Particle> new_particles = {};
    for(int i = 0; i < num_particles; i++){
        int k = dist(gen);
        Particle new_particle = {};

        new_particle.id = i;
        new_particle.x = particles[k].x;
        new_particle.y = particles[k].y;
        new_particle.theta = particles[k].theta;
        new_particle.weight = particles[k].weight;
        new_particles.push_back(new_particle);
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
