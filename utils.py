from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from scipy.spatial.transform import Rotation

def get_omega_imu(q, omega):
    rot_ = Rotation.from_quat(q)
    mat_ = rot_.as_matrix()
    vec_ = np.dot(np.transpose(mat_), np.array([[x] for x in omega]))
    return np.reshape(vec_,(-1,))

def get_gravity_vec(q):
    rot_ = Rotation.from_quat(q)
    mat_ = rot_.as_matrix()
    vec_ = np.dot(np.transpose(mat_), np.array([[0.], [0.], [-1.]]))
    out_ = np.reshape(vec_, (-1,))
    return out_

def rbf_reward(x,xhat,alpha):
    x = np.array(x)
    xhat = np.array(xhat)
    return np.exp(alpha * np.sum(np.square(x-xhat)))

class AutoSaveCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, n_procs, save_model_dir, verbose=1):
        super(AutoSaveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.n_procs = n_procs
        self.save_path = os.path.join(save_model_dir, 'auto_save/')
        self.best_mean_reward = -np.inf

    # def _init_callback(self) -> None:
    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    # def _on_step(self) -> bool:
    def _on_step(self):
        if self.n_calls * self.n_procs % self.check_freq == 0:
            print('self.n_calls: ',self.n_calls)
            model_path1 = os.path.join(self.save_path, 'model_{}'.format(self.n_calls * self.n_procs))
            self.model.save(model_path1)

        return True