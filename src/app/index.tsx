import { useRouter } from 'expo-router';
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

export default function HomeScreen() {
  const router = useRouter();
  const theme = useTheme();

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.background }]}>
      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
      >
        {/* Header Branding */}
        <View style={styles.header}>
          <View style={styles.brandRow}>
            <View style={styles.logoBadge}>
              <Text style={styles.logoBadgeText}>🩺</Text>
            </View>
            <View>
              <ThemedText style={styles.brandTitle}>Nivora</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                AI Digital Biomarkers & Clinical Screening Suite
              </ThemedText>
            </View>
          </View>
        </View>

        {/* 1. Jaundice Screening Card */}
        <ThemedView type="backgroundElement" style={[styles.screeningCard, { borderColor: '#F59E0B' }]}>
          <View style={styles.badgeRow}>
            <View style={[styles.liveTag, { backgroundColor: '#F59E0B' }]}>
              <Text style={styles.liveTagText}>JAUNDICE & SCLERA</Text>
            </View>
            <ThemedText type="small" themeColor="textSecondary">
              TFLite Model
            </ThemedText>
          </View>

          <ThemedText style={styles.featuredTitle}>Jaundice & Bilirubin Screening</ThemedText>
          <ThemedText style={styles.featuredDesc} themeColor="textSecondary">
            Non-invasive quantification of scleral icterus and transcutaneous bilirubin from eye/facial photos using `jaundice_model.tflite`.
          </ThemedText>

          <View style={styles.metricHighlightsRow}>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillLabel}>Sclera Colorimetry</Text>
              <Text style={styles.metricPillVal}>Yellow-to-Blue Ratio</Text>
            </View>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillLabel}>Serum Bilirubin</Text>
              <Text style={styles.metricPillVal}>mg/dL Estimate</Text>
            </View>
          </View>

          <Pressable
            style={({ pressed }) => [styles.primaryActionBtn, { backgroundColor: '#D97706' }, pressed && styles.pressed]}
            onPress={() => router.push('/jaundice' as any)}
          >
            <Text style={styles.primaryActionText}>Start Jaundice Test 🟡 →</Text>
          </Pressable>
        </ThemedView>

        {/* 2. Cataract Screening Card */}
        <ThemedView type="backgroundElement" style={[styles.screeningCard, { borderColor: '#0284C7' }]}>
          <View style={styles.badgeRow}>
            <View style={[styles.liveTag, { backgroundColor: '#0284C7' }]}>
              <Text style={styles.liveTagText}>CATARACT & OPHTHALMIC</Text>
            </View>
            <ThemedText type="small" themeColor="textSecondary">
              Float16 TFLite
            </ThemedText>
          </View>

          <ThemedText style={styles.featuredTitle}>Cataract & Lens Opacity AI</ThemedText>
          <ThemedText style={styles.featuredDesc} themeColor="textSecondary">
            Deep learning anterior segment analysis for crystalline lens clouding, nuclear sclerosis, and pupil opacity screening using `cataract_detector_float16.tflite`.
          </ThemedText>

          <View style={styles.metricHighlightsRow}>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillLabel}>Lens Opacity</Text>
              <Text style={styles.metricPillVal}>Grade 0-4</Text>
            </View>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillLabel}>Model Accuracy</Text>
              <Text style={styles.metricPillVal}>Float16 Neural Net</Text>
            </View>
          </View>

          <Pressable
            style={({ pressed }) => [styles.primaryActionBtn, { backgroundColor: '#0284C7' }, pressed && styles.pressed]}
            onPress={() => router.push('/cataract' as any)}
          >
            <Text style={styles.primaryActionText}>Start Cataract Test 👁️ →</Text>
          </Pressable>
        </ThemedView>

        {/* 3. Anemia Screening Card */}
        <ThemedView type="backgroundElement" style={[styles.screeningCard, { borderColor: '#EC4899' }]}>
          <View style={styles.badgeRow}>
            <View style={[styles.liveTag, { backgroundColor: '#EC4899' }]}>
              <Text style={styles.liveTagText}>ANEMIA & PALLOR</Text>
            </View>
            <ThemedText type="small" themeColor="textSecondary">
              Colorimetric AI
            </ThemedText>
          </View>

          <ThemedText style={styles.featuredTitle}>Anemia & Hemoglobin Screening</ThemedText>
          <ThemedText style={styles.featuredDesc} themeColor="textSecondary">
            Palpebral conjunctival micro-vascularization and erythema index (EI) analysis for non-invasive hemoglobin estimation and pallor screening.
          </ThemedText>

          <View style={styles.metricHighlightsRow}>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillLabel}>Conjunctiva</Text>
              <Text style={styles.metricPillVal}>Erythema Index</Text>
            </View>
            <View style={styles.metricPill}>
              <Text style={styles.metricPillLabel}>Hemoglobin</Text>
              <Text style={styles.metricPillVal}>g/dL WHO Range</Text>
            </View>
          </View>

          <Pressable
            style={({ pressed }) => [styles.primaryActionBtn, { backgroundColor: '#BE185D' }, pressed && styles.pressed]}
            onPress={() => router.push('/anemia' as any)}
          >
            <Text style={styles.primaryActionText}>Start Anemia Test 🩸 →</Text>
          </Pressable>
        </ThemedView>

        {/* 4. Secondary: Voice & Motor Neurological Screening */}
        <ThemedView type="backgroundElement" style={styles.secondaryCard}>
          <View style={styles.badgeRow}>
            <View style={[styles.liveTag, { backgroundColor: '#64748B' }]}>
              <Text style={styles.liveTagText}>NEUROLOGICAL MODULE</Text>
            </View>
          </View>
          <ThemedText style={styles.secondaryTitle}>Voice Phonation & Tremor Screening</ThemedText>
          <ThemedText style={styles.secondaryDesc} themeColor="textSecondary">
            Acoustic micro-tremor and vocal dysphonia analysis for motor/neurological tracking.
          </ThemedText>
          <Pressable
            style={({ pressed }) => [styles.outlineActionBtn, pressed && styles.pressed]}
            onPress={() => router.push('/parkinsons')}
          >
            <Text style={styles.outlineActionText}>Open Neurological Module 🎙️ →</Text>
          </Pressable>
        </ThemedView>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  scrollContainer: {
    paddingHorizontal: Spacing.four,
    paddingVertical: Spacing.four,
    paddingBottom: BottomTabInset + 40,
    maxWidth: MaxContentWidth,
    alignSelf: 'center',
    width: '100%',
    gap: Spacing.four,
  },
  header: {
    paddingBottom: Spacing.two,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.three,
  },
  logoBadge: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#0284C7',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#0284C7',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 6,
    elevation: 3,
  },
  logoBadgeText: {
    fontSize: 22,
  },
  brandTitle: {
    fontSize: 22,
    fontWeight: '800',
    letterSpacing: -0.3,
  },
  screeningCard: {
    borderRadius: 16,
    padding: Spacing.four,
    borderWidth: 1.5,
    gap: Spacing.three,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  badgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  liveTag: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  liveTagText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  featuredTitle: {
    fontSize: 19,
    fontWeight: '700',
    letterSpacing: -0.2,
  },
  featuredDesc: {
    fontSize: 13,
    lineHeight: 19,
  },
  metricHighlightsRow: {
    flexDirection: 'row',
    gap: Spacing.two,
  },
  metricPill: {
    flex: 1,
    padding: Spacing.two,
    borderRadius: 8,
    backgroundColor: '#F8FAFC',
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  metricPillLabel: {
    fontSize: 11,
    color: '#64748B',
    fontWeight: '500',
  },
  metricPillVal: {
    fontSize: 12,
    fontWeight: '700',
    color: '#0F172A',
    marginTop: 2,
  },
  primaryActionBtn: {
    paddingVertical: 13,
    borderRadius: 10,
    alignItems: 'center',
  },
  primaryActionText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },
  secondaryCard: {
    borderRadius: 14,
    padding: Spacing.four,
    borderWidth: 1,
    borderColor: '#E2E8F0',
    gap: Spacing.two,
  },
  secondaryTitle: {
    fontSize: 16,
    fontWeight: '700',
  },
  secondaryDesc: {
    fontSize: 13,
    lineHeight: 18,
  },
  outlineActionBtn: {
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#CBD5E1',
    alignItems: 'center',
    marginTop: 4,
  },
  outlineActionText: {
    color: '#334155',
    fontSize: 13,
    fontWeight: '700',
  },
  pressed: {
    opacity: 0.8,
    transform: [{ scale: 0.98 }],
  },
});
