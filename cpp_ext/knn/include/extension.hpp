# pragma once
#include "nanoflann.hpp"

/**
 * A result-set class used when performing a radius based search.
 * modifified based on RadiusResultSet. Only takes K results instead of exhaust all pionts within the radius
 * "ball query"
 */
template <typename DistanceType, typename IndexType = size_t>
class KRadiusResultSet
{
public:
    const DistanceType radius;
    const size_t nk;	// number of neighbors
    size_t counter;		// counter of good results

    std::vector<std::pair<IndexType, DistanceType> > &m_indices_dists;

    inline KRadiusResultSet(DistanceType radius_, size_t nk_, std::vector<std::pair<IndexType,DistanceType> > &indices_dists) : radius(radius_), nk(nk_), m_indices_dists(indices_dists)
    {
        counter = 0;
        init();
    }

    inline void init() {clear(); m_indices_dists.reserve(nk);}
    inline void clear() { m_indices_dists.clear(); counter=0;}

    inline size_t size() const { return nk;}

    inline bool full() const { return counter < nk; }

            /**
             * Called during search to add an element matching the criteria.
             * @return true if the search should be continued, false if the results are sufficient
             */
            inline bool addPoint(DistanceType dist, IndexType index)
    {
        if (dist < radius && counter < nk){
            if (counter == 0){
                for (size_t i=0; i<nk; ++i)
                    m_indices_dists.push_back(std::make_pair(index, dist));	// duplicate the first	
            } else {
                m_indices_dists[counter] = std::make_pair(index, dist);		// overwrite
            }
            counter++;
        }
        if (counter < nk)
            return true;
        else
            return false;
    }

    inline DistanceType worstDist() const { return radius; }

    /**
     * Find the worst result (furtherest neighbor) without copying or sorting
     * Pre-conditions: size() > 0
     */
    std::pair<IndexType,DistanceType> worst_item() const
    {
        if (m_indices_dists.empty()) throw std::runtime_error("Cannot invoke RadiusResultSet::worst_item() on an empty list of results.");
        typedef typename std::vector<std::pair<IndexType, DistanceType> >::const_iterator DistIt;
        DistIt it = std::max_element(m_indices_dists.begin(), m_indices_dists.end(), nanoflann::IndexDist_Sorter());
        return *it;
    }
};


