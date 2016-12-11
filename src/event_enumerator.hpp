template<class T>
class EventEnumerator {
  public:
    class iterator {
      friend class EventEnumerator;

      private:
        unsigned int i_curr;

      protected:
        iterator(int i) : i_curr(i) {}

      public:
        T operator*() const { return T(i_curr); }

        iterator& operator++() {
          i_curr++;
          return *this;
        }

        iterator operator++(int) {
          iterator copy(*this);
          i_curr++;
          return copy;
        }

        bool operator==(const iterator &other) const {
          return i_curr == other.i_curr;
        }

        bool operator!=(const iterator &other) const {
          return i_curr != other.i_curr;
        }
    };

  iterator begin() const { return begin_it; }
  iterator end() const { return end_it; }
  EventEnumerator() : begin_it(0), end_it(T::cardinality) {}

  private:
    iterator begin_it;
    iterator end_it;
};

