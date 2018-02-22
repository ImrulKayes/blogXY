from scripts.crawler.friends import get_friends
from scripts.crawler.profile import get_blog_statistics
import conf.config as conf

from Queue import Queue
from threading import Thread, Lock
import logging


class BFSCrwaler(Thread):
    """ Breadth first search (BFS) crawler to collect blogster platform data.
        This crawler initiates a queue reading a list of seed bloggers from blogster.
        It runs BFS algorithm and iteratively discovers bloggers' friends and collects their blog related data.
    """
    def __init__(self):
        Thread.__init__(self)
        # initiates a queue for blogger to be processed
        self.queue = Queue()

        # a dictionary keeps tracks which bloggers already processed
        self.visited_nodes_dic = {}

        # lock for the visited_nodes_dic dictionary
        self.visited_node_lock = Lock()
        self.output_file_lock = Lock()

        # read the number of threads to be used in parallel from config
        self.MAX_THREADS = conf.thread_num

        # create output files where threads will write their data
        self.profile_data_file = open(conf.profile_data_file, "wb")
        self.edge_data_file = open(conf.edge_data_file, "wb")

        # id for row/serial number when writing the data
        self.id = 0

        # set logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(conf.crawler_log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def process(self):
        # read the seed blogger
        self.insert_seed_blogger(conf.seed_blogger_file)

        # start the crawling threads
        self.process_threads = [Thread(target = self.process_bloggers, args = (i,)) for i in range(self.MAX_THREADS)]
        self.logger.info("initialized threas")
        map(lambda thread: thread.start(), self.process_threads)
        map(lambda thread: thread.join(), self.process_threads)

        # close the files
        self.profile_data_file.close()
        self.edge_data_file.close()

    def insert_seed_blogger(self, seed_blogger_filename):
        """ Reads the seed bloggers and put them in a queue"""

        for blogger in open(seed_blogger_filename):
            self.queue.put(blogger.strip())
            self.visited_nodes_dic[blogger] = True
        self.logger.info("Loaded seed blogger in queue")

    def process_bloggers(self, index):
        """Starts threads and runs the BFS"""
        self.logger.info("Starting thread {0}".format(index))

        # run the thread until the queue is empty
        while not self.queue.empty():

            # get a blogger from the queue
            user = self.queue.get()
            self.logger.info("Queue size: {0}, processing blogger {1}".format(self.queue.qsize(), user))
            try:
                # get the blogger's profile and friends data
                profile_data = get_blog_statistics(user, self.logger)
                all_friends = get_friends(user, self.logger)

                if profile_data and all_friends:

                    # write profile and friendship data
                    blogger = str(profile_data[0])
                    profile_data_str = '|'.join(map(lambda x: str(x), profile_data))
                    blogger_fnds_str = '|'.join([blogger] + map(lambda x: str(x), all_friends))
                    self.write_data(profile_data_str, blogger_fnds_str)

                    # if we haven't process a friend of the blogger put the friend in the queue
                    for friend in all_friends:
                        self.visited_node_lock.acquire()
                        if not self.visited_nodes_dic.has_key(friend):
                            self.visited_nodes_dic[friend] = True
                            self.queue.put(friend)
                        self.visited_node_lock.release()
            except Exception as e:
                print "Exception user: {0}".format(user)
                print e.message
                raise
        self.logger.info("Exiting thread {0}".format(index))

    def write_data(self, profile_data_str, blogger_fnds_str):
        """Writes a blogger's profile and friendship data in files"""
        self.output_file_lock.acquire()
        self.id += 1
        self.profile_data_file.write(str(self.id)+"|"+profile_data_str+"\n")
        self.edge_data_file.write(blogger_fnds_str+"\n")
        self.output_file_lock.release()

if __name__ == '__main__':
    """Entry of the main function"""
    bfs_crawler = BFSCrwaler()
    bfs_crawler.process()


