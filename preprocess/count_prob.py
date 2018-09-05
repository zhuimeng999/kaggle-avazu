from FE.count_feature import output_dir, train_filename
import logging
import os, sys
from datetime import datetime
import collections
import pickle

logger = logging.getLogger('FE')
if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    hour_prob = collections.defaultdict(lambda: [0, 0])
    weekday_prob = collections.defaultdict(lambda: [0, 0])
    site_prob = collections.defaultdict(lambda: [0, 0])
    app_prob = collections.defaultdict(lambda: [0, 0])
    ip_prob = collections.defaultdict(lambda: [0, 0])
    id_prob = collections.defaultdict(lambda: [0, 0])

    logger.info('build progress %d', 0)
    with open(os.path.join(os.path.dirname(__file__), '../output/train_split.csv'), 'r') as f:
        header = f.readline().strip().split(',')
        for line_no, line in enumerate(f):
            features = line.strip().split(',')
            parsed_time = datetime.strptime('20' + features[2], '%Y%m%d%H')
            hour_prob[parsed_time.hour][0] += 1
            weekday_prob[parsed_time.weekday()][0] += 1
            site_prob[features[5]][0] += 1
            app_prob[features[8]][0] += 1
            id_prob[features[11]][0] += 1
            ip_prob[features[12]][0] += 1

            if features[1] == '1':
                hour_prob[parsed_time.hour][1] += 1
                weekday_prob[parsed_time.weekday()][1] += 1
                site_prob[features[5]][1] += 1
                app_prob[features[8]][1] += 1
                id_prob[features[11]][1] += 1
                ip_prob[features[12]][1] += 1
            elif features[1] == '0':
                pass
            else:
                assert False, features
            if (line_no % 400000) == 0:
                logger.info('build progress %d', line_no)
        logger.info('build progress %d, done!!!', line_no)

    prob_map_dict = {
        'hour': dict(hour_prob),
        'weekday': dict(weekday_prob),
        'site_id': dict(site_prob),
        'app_id': dict(app_prob),
        'device_id': dict(id_prob),
        'device_ip': dict(ip_prob)
    }

    with open(os.path.join(output_dir, 'prob_map.pickle'), 'wb') as f:
        pickle.dump(prob_map_dict, f, pickle.HIGHEST_PROTOCOL)
