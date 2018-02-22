import urllib2
import re
from BeautifulSoup import BeautifulSoup
import numpy as np
import conf.config as conf

def get_blog_statistics(user, logger):
    """Collects blog statistics of blogger and returns as a list"""

    # url to crawl
    page = "http://www.blogster.com/"+user
    try:
        soup = BeautifulSoup(urllib2.urlopen(page))
    except Exception as e:
        logger.error(e.message)
        logger.error("Error: {0}".format(page))
        return []

    user_statistics_data=[np.nan for i in range(len(conf.blogster_profile_schema)-1)]
    user_statistics_data[0] = user

    try:
        if soup.find("div", {"id": "main-content"}).h1.renderContents() == "This area is private.":
            user_statistics_data[1] = "Community"
        elif soup.find("div", {"id": "main-content"}).h1.renderContents() == "This profile is private.":
                user_statistics_data[1] = "Private"
        else:
            user_statistics_data[1] = "Public"
            for i in soup.findAll('div', {'class': 'tabbox1-page left-menu-stats'}):
                for j in i.findAll('p'):

                    rawstr = str(j)
                    if re.search(r'<p><span class=\"lfloat\">Birthday:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Birthday:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[2] = (re.sub(r'</span></p>','',raw1)).strip().replace("|", ",")

                    if re.search(r'<p><span class=\"lfloat\">Gender:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Gender:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[3] = (re.sub(r'</span></p>','',raw1)).strip().strip().replace("|", ",")

                    if re.search(r'<p><span class=\"lfloat\">Home:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Home:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[4] = (re.sub(r'</span></p>','',raw1)).strip().replace("|", ",")

                    if re.search(r'<p><span class=\"lfloat\">Status:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Status:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[5] = (re.sub(r'</span></p>','',raw1)).strip().replace("|", ",")


                    if re.search(r'<p><span class=\"lfloat\">Joined:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Joined:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[6] = (re.sub(r'</span></p>','',raw1)).strip().replace("|", ",")

                    if re.search(r'<p><span class=\"lfloat\">Job:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Job:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[7] = (re.sub(r'</span></p>','',raw1)).strip().replace("|", ",")

                    if re.search(r'<p><span class=\"lfloat\">Language:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Language:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[8] = (re.sub(r'</span></p>','',raw1)).strip().replace("|", ",")

                    # This is blogStat
                    if re.search(r'<p><span class=\"lfloat\">Blog Traffic:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Blog Traffic:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[9] = (re.sub(r'</span></p>','',raw1)).strip()

                    if re.search(r'<p><span class=\"lfloat\">Posts:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Posts:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[10] = (re.sub(r'</span></p>','',raw1)).strip()

                    if re.search(r'<p><span class=\"lfloat\">My Comments:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">My Comments:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[11] = (re.sub(r'</span></p>','',raw1)).strip()

                    if re.search(r'<p><span class=\"lfloat\">User Comments:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">User Comments:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[12] = (re.sub(r'</span></p>','',raw1)).strip()

                    if re.search(r'<p><span class=\"lfloat\">Photos:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Photos:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[13] = (re.sub(r'</span></p>','',raw1)).strip()

                    if re.search(r'<p><span class=\"lfloat\">Friends:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Friends:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[14] = (re.sub(r'</span></p>','',raw1)).strip()

                    if re.search(r'<p><span class=\"lfloat\">Following:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Following:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[15] = (re.sub(r'</span></p>','',raw1)).strip()

                    if re.search(r'<p><span class=\"lfloat\">Followers:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Followers:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[16] = (re.sub(r'</span></p>','',raw1)).strip()

                    if re.search(r'<p><span class=\"lfloat\">Points:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Points:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[17] = (re.sub(r'</span></p>','',raw1)).strip()

                    if re.search(r'<p><span class=\"lfloat\">Last Online:</span> <span class=\"rfloat\">',rawstr):
                        raw1 = re.sub(r'<p><span class=\"lfloat\">Last Online:</span> <span class=\"rfloat\">','',rawstr)
                        user_statistics_data[18] = (re.sub(r'</span></p>','',raw1)).strip().replace("|", ",")

        return user_statistics_data
    except Exception as e:
        logger.error("Exception {0}, Didn't get data for {1}".format(e.message, user))
        logger.error("Error: page {0}".format(page))
        return []












